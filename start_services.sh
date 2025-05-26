#!/bin/bash

export LD_LIBRARY_PATH="/nix/store/qxfi4d8dfc8rpdk3y0dlmdc28nad02pd-zlib-1.2.13/lib:/nix/store/22nxhmsfcv2q2rpkmfvzwg2w5z1l231z-gcc-13.3.0-lib/lib:$LD_LIBRARY_PATH"

# Ensure we're using PostgreSQL 14
export PATH="/nix/store/0rcrmxk4y6w0gl96a2nzjb78gv8r8vyv-postgresql-14.11/bin:$PATH"

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    set -o allexport
    source .env
    set +o allexport
fi

# Function to check if we're running on Replit
is_replit() {
    [ -n "$REPL_ID" ]
}

# Set PostgreSQL environment variables based on environment
if is_replit; then
    # On Replit, use the 'runner' user which already exists
    CURRENT_USER=$(whoami)
    echo "Running on Replit, using user: $CURRENT_USER"
    
    export POSTGRES_USER="$CURRENT_USER"
    export POSTGRES_HOST="localhost"
    export POSTGRES_PORT=5432
    export POSTGRES_DB="transparentsf"
    export POSTGRES_PASSWORD=""
else
    # On other environments, use .env settings or defaults
    if [ -z "$POSTGRES_USER" ]; then
        export POSTGRES_USER="postgres"
    fi
    if [ -z "$POSTGRES_HOST" ]; then
        export POSTGRES_HOST="localhost"
    fi
    if [ -z "$POSTGRES_PORT" ]; then
        export POSTGRES_PORT=5432
    fi
    if [ -z "$POSTGRES_DB" ]; then
        export POSTGRES_DB="transparentsf"
    fi
    if [ -z "$POSTGRES_PASSWORD" ]; then
        export POSTGRES_PASSWORD=""
    fi
fi

# Set standard PostgreSQL environment variables for CLI tools
export PGUSER="$POSTGRES_USER"
export PGHOST="$POSTGRES_HOST"
export PGPORT="$POSTGRES_PORT"
export PGDATABASE="$POSTGRES_DB"
export PGPASSWORD="$POSTGRES_PASSWORD"

echo "PostgreSQL connection settings:"
echo "  POSTGRES_USER: $POSTGRES_USER"
echo "  POSTGRES_HOST: $POSTGRES_HOST"
echo "  POSTGRES_DB: $POSTGRES_DB"

# Function to wait for a service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=60
    local attempt=1
    
    echo "Waiting for $name to be ready..."
    while ! curl -s "$url" > /dev/null 2>&1; do
        if [ $attempt -ge $max_attempts ]; then
            echo "$name failed to start after $max_attempts attempts"
            exit 1
        fi
        echo "Attempt $attempt: $name not ready yet..."
        sleep 3
        ((attempt++))
    done
    echo "$name is ready!"
}

# Function to create the database if it doesn't exist
create_database() {
    echo "Checking if database '$POSTGRES_DB' exists..."
    if ! psql -h $POSTGRES_HOST -U $POSTGRES_USER -lqt | cut -d \| -f 1 | grep -qw $POSTGRES_DB; then
        echo "Creating database '$POSTGRES_DB'..."
        createdb -h $POSTGRES_HOST -U $POSTGRES_USER $POSTGRES_DB
        echo "Database created successfully."
        return 0  # Database was just created
    else
        echo "Database '$POSTGRES_DB' already exists."
        return 1  # Database already existed
    fi
}

start_postgres() {
    # Check for PostgreSQL version
    PG_VERSION=$(postgres --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
    echo "Using PostgreSQL version: $PG_VERSION"
    
    if is_replit; then
        echo "Setting up PostgreSQL on Replit..."

        # Check if PostgreSQL is already running
        if pgrep -f "postgres -D" > /dev/null; then
            echo "PostgreSQL is already running"
            
            # Check PostgreSQL connection
            echo "Checking PostgreSQL connection..."
            max_attempts=30
            attempt=1
            while ! pg_isready -p 5432 -h localhost > /dev/null 2>&1; do
                if [ $attempt -ge $max_attempts ]; then
                    echo "Cannot connect to PostgreSQL after $max_attempts attempts"
                    exit 1
                fi
                echo "Attempt $attempt: Cannot connect to PostgreSQL yet..."
                sleep 2
                ((attempt++))
            done
            echo "PostgreSQL connection successful!"
            
            # Try to connect and get database info
            echo "Getting PostgreSQL info..."
            psql -h localhost -c "\\conninfo" 2>/dev/null || echo "Failed to get connection info"
            
            # Create the database if it doesn't exist
            create_database
            
            return
        fi

        echo "Using home directory for PostgreSQL data on Replit"
        export PGDATA=$HOME/postgres_data # Always use persistent storage

        # Create data directory and initialize if needed
        if [ ! -d "$PGDATA/base" ]; then
            echo "Initializing PostgreSQL data directory at $PGDATA..."
            mkdir -p "$PGDATA" # Ensure the parent PGDATA directory exists
            initdb -D "$PGDATA"
            
            CONF_FILE="$PGDATA/postgresql.conf"
            echo "Configuring PostgreSQL socket and PID locations in $CONF_FILE..."

            # Ensure unix_socket_directories is set correctly
            if grep -q "^#*unix_socket_directories" "$CONF_FILE"; then
                sed -i "s|^#*unix_socket_directories.*|unix_socket_directories = '$HOME'|" "$CONF_FILE"
            else
                echo "unix_socket_directories = '$HOME'" >> "$CONF_FILE"
            fi
            
            # Ensure external_pid_file is set correctly
            if grep -q "^#*external_pid_file" "$CONF_FILE"; then
                sed -i "s|^#*external_pid_file.*|external_pid_file = '$HOME/postgres.pid'|" "$CONF_FILE"
            else
                echo "external_pid_file = '$HOME/postgres.pid'" >> "$CONF_FILE"
            fi
        else
            echo "Using existing PostgreSQL data directory at $PGDATA"
        fi

        # Check for stale pid file
        if [ -f "$PGDATA/postmaster.pid" ]; then
            PID=$(head -1 "$PGDATA/postmaster.pid")
            if ! ps -p "$PID" > /dev/null; then
                echo "Removing stale PID file..."
                rm "$PGDATA/postmaster.pid"
            fi
        fi

        echo "Starting PostgreSQL..."
        pg_ctl -D "$PGDATA" -l "$HOME/postgres_log" start

        echo "Waiting for PostgreSQL to be ready..."
        max_attempts=30
        attempt=1
        while ! pg_isready -p 5432 -h localhost > /dev/null 2>&1; do
            if [ $attempt -ge $max_attempts ]; then
                echo "PostgreSQL failed to start after $max_attempts attempts"
                exit 1
            fi
            echo "Attempt $attempt: PostgreSQL not ready yet..."
            sleep 2
            ((attempt++))
        done
        echo "PostgreSQL is ready!"

        # Create the database if it doesn't exist
        create_database

    else
        # For macOS, just check if PostgreSQL is running
        echo "Checking PostgreSQL on macOS..."
        if ! pg_isready -h localhost > /dev/null 2>&1; then
            echo "PostgreSQL is not running. Please start it with: brew services start postgresql"
            exit 1
        fi
        echo "PostgreSQL is running."
        
        # Create the database if it doesn't exist
        create_database
    fi
}

# Function to initialize the database
init_database() {
    echo "Checking database status..."
    
    if [ -d "ai" ]; then
        cd ai
        
        # Check if tables exist by directly using psql
        echo "Checking if database tables exist..."
        TABLE_COUNT=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'")
        
        # Trim whitespace
        TABLE_COUNT=$(echo $TABLE_COUNT | tr -d ' ')
        
        if [ "$TABLE_COUNT" -eq "0" ]; then
            echo "No tables found in database. Running initialization..."
            python tools/init_postgres_db.py
            echo "Database initialization completed."
        else
            echo "Database tables already exist. Skipping initialization."
        fi
        
        cd ..
    else
        echo "Error: 'ai' directory not found"
        exit 1
    fi
}

# Check for Node.js
check_node() {
    if ! command -v node &> /dev/null; then
        echo "Node.js is not installed. Ghost Bridge will not start."
        echo "To install Node.js, please run: curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash - && sudo apt-get install -y nodejs"
        return 1
    else
        echo "Node.js is installed: $(node --version)"
        return 0
    fi
}

# Check for Qdrant
check_qdrant() {
    if ! command -v qdrant &> /dev/null; then
        echo "Qdrant is not installed. Vector database will not be available."
        echo "To install Qdrant, please follow the instructions at: https://qdrant.tech/documentation/install/"
        return 1
    else
        echo "Qdrant is installed"
        return 0
    fi
}

# Kill any lingering processes
pkill -f qdrant || true
pkill -f main.py || true
pkill -f ghostBridge.js || true

# Create logs directory
mkdir -p ai/logs

# Start PostgreSQL and initialize DB
start_postgres
init_database

# Start backend
echo "Starting Main..."
cd ai
python main.py &
cd ..

wait_for_service "http://0.0.0.0:8000/backend" "Backend"

# Confirm frontend
echo "Checking frontend..."
wait_for_service "http://0.0.0.0:8000" "Frontend"

# Start Qdrant
echo "Starting Qdrant..."
if check_qdrant; then
    qdrant &
    wait_for_service "http://0.0.0.0:6333/healthz" "Qdrant"
else
    echo "Continuing without Qdrant..."
fi

# Start Ghost Bridge
echo "Starting Ghost Bridge..."
if check_node; then
    cd ai
    # Check if Ghost Bridge configuration exists
    if [ -f .env ] && grep -q "GHOST_URL" .env; then
        echo "Found Ghost configuration, starting Ghost Bridge..."
        node tools/ghost_bridge/ghostBridge.js &
    else
        echo "Ghost Bridge configuration is missing. Please add GHOST_URL to your .env file."
    fi
    cd ..
fi

# Hold the script open
wait

echo "Services started. Check logs/ directory for output."
