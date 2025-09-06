#!/bin/bash

export LD_LIBRARY_PATH="/nix/store/qxfi4d8dfc8rpdk3y0dlmdc28nad02pd-zlib-1.2.13/lib:/nix/store/22nxhmsfcv2q2rpkmfvzwg2w5z1l231z-gcc-13.3.0-lib/lib:$LD_LIBRARY_PATH"

# Use system PostgreSQL (don't force version 14)

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

# Function to check if we should use managed database
use_managed_db() {
    [ -n "$DATABASE_URL" ]
}

# Set PostgreSQL environment variables based on environment
if use_managed_db; then
    echo "Using managed database with DATABASE_URL"
    echo "Individual PostgreSQL environment variables will be ignored"
    # Don't override individual variables if DATABASE_URL is set
    # The db_utils.py will handle the connection using DATABASE_URL
elif is_replit; then
    # On Replit without managed DB, use local PostgreSQL
    CURRENT_USER=$(whoami)
    echo "Running on Replit with local PostgreSQL, using user: $CURRENT_USER"
    
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

# Set standard PostgreSQL environment variables for CLI tools (only if not using DATABASE_URL)
if use_managed_db; then
    echo "Database connection via DATABASE_URL (managed database)"
    echo "  DATABASE_URL: [REDACTED for security]"
else
    export PGUSER="$POSTGRES_USER"
    export PGHOST="$POSTGRES_HOST"
    export PGPORT="$POSTGRES_PORT"
    export PGDATABASE="$POSTGRES_DB"
    export PGPASSWORD="$POSTGRES_PASSWORD"

    echo "PostgreSQL connection settings:"
    echo "  POSTGRES_USER: $POSTGRES_USER"
    echo "  POSTGRES_HOST: $POSTGRES_HOST"
    echo "  POSTGRES_DB: $POSTGRES_DB"
fi

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

# Function to create the database if it doesn't exist (only for local PostgreSQL)
create_database() {
    if use_managed_db; then
        echo "Using managed database - skipping database creation check"
        return 1  # Assume database exists in managed environment
    fi
    
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
        export PGDATA=./postgres_data # Always use persistent storage

        # Create data directory and initialize if needed
        if [ ! -d "$PGDATA/base" ]; then
            echo "Initializing PostgreSQL data directory at $PGDATA..."
            # Remove any existing corrupted data directory
            if [ -d "$PGDATA" ]; then
                echo "Removing corrupted PostgreSQL data directory..."
                rm -rf "$PGDATA"
            fi
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
            echo "Checking existing PostgreSQL data directory at $PGDATA..."
            # Check if the data directory is compatible with current PostgreSQL version
            NEEDS_REINIT=false
            
            # First check if it's a valid PostgreSQL cluster
            if ! pg_controldata "$PGDATA" > /dev/null 2>&1; then
                echo "Existing PostgreSQL data directory is corrupted, reinitializing..."
                NEEDS_REINIT=true
            else
                # Check PostgreSQL version compatibility
                if [ -f "$PGDATA/PG_VERSION" ]; then
                    DATA_VERSION=$(cat "$PGDATA/PG_VERSION")
                    CURRENT_MAJOR=$(echo "$PG_VERSION" | cut -d. -f1)
                    DATA_MAJOR=$(echo "$DATA_VERSION" | cut -d. -f1)
                    
                    if [ "$CURRENT_MAJOR" != "$DATA_MAJOR" ]; then
                        echo "PostgreSQL version mismatch: data directory is version $DATA_VERSION, but running version $PG_VERSION"
                        echo "Major version upgrade required, reinitializing data directory..."
                        NEEDS_REINIT=true
                    else
                        echo "PostgreSQL versions compatible: data=$DATA_VERSION, current=$PG_VERSION"
                    fi
                else
                    echo "Cannot determine data directory PostgreSQL version, reinitializing..."
                    NEEDS_REINIT=true
                fi
            fi
            
            if [ "$NEEDS_REINIT" = true ]; then
                echo "Backing up and reinitializing PostgreSQL data directory..."
                if [ -d "$PGDATA.backup" ]; then
                    rm -rf "$PGDATA.backup"
                fi
                mv "$PGDATA" "$PGDATA.backup" 2>/dev/null || rm -rf "$PGDATA"
                mkdir -p "$PGDATA"
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
        if ! pg_ctl -D "$PGDATA" -l "$HOME/postgres_log" start; then
            echo "Failed to start PostgreSQL. Checking log file..."
            if [ -f "$HOME/postgres_log" ]; then
                echo "=== PostgreSQL Log Output ==="
                tail -20 "$HOME/postgres_log"
                echo "=== End Log Output ==="
            fi
            echo "Attempting to reinitialize PostgreSQL data directory due to startup failure..."
            
            # Force reinitialize if startup failed
            rm -rf "$PGDATA"
            mkdir -p "$PGDATA"
            initdb -D "$PGDATA"
            
            CONF_FILE="$PGDATA/postgresql.conf"
            echo "Reconfiguring PostgreSQL socket and PID locations..."
            
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
            
            echo "Attempting to start PostgreSQL after reinitialization..."
            if ! pg_ctl -D "$PGDATA" -l "$HOME/postgres_log" start; then
                echo "PostgreSQL still failed to start after reinitialization. Check logs:"
                if [ -f "$HOME/postgres_log" ]; then
                    cat "$HOME/postgres_log"
                fi
                exit 1
            fi
        fi

        echo "Waiting for PostgreSQL to be ready..."
        max_attempts=30
        attempt=1
        while ! pg_isready -p 5432 -h localhost > /dev/null 2>&1; do
            if [ $attempt -ge $max_attempts ]; then
                echo "PostgreSQL failed to be ready after $max_attempts attempts"
                if [ -f "$HOME/postgres_log" ]; then
                    echo "=== Latest PostgreSQL Log Output ==="
                    tail -20 "$HOME/postgres_log"
                    echo "=== End Log Output ==="
                fi
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
        
        # Check if tables exist using our centralized db_utils
        echo "Checking if database tables exist..."
        
        # Create a simple Python script to test database connection and table count
        python3 -c "
import sys
sys.path.append('.')
from tools.db_utils import get_postgres_connection
import psycopg2.extras

try:
    conn = get_postgres_connection()
    if conn is None:
        print('ERROR: Could not connect to database')
        sys.exit(1)
    
    cursor = conn.cursor()
    cursor.execute(\"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'\")
    table_count = cursor.fetchone()[0]
    print(f'TABLES: {table_count}')
    
    cursor.close()
    conn.close()
    sys.exit(0)
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
" > /tmp/db_check_output.txt 2>&1

        # Check the result
        if grep -q "ERROR:" /tmp/db_check_output.txt; then
            echo "Database connection failed:"
            cat /tmp/db_check_output.txt
            echo "Exiting due to database connection failure"
            exit 1
        fi
        
        TABLE_COUNT=$(grep "TABLES:" /tmp/db_check_output.txt | cut -d' ' -f2)
        
        if [ "$TABLE_COUNT" -eq "0" ] 2>/dev/null; then
            echo "No tables found in database. Running initialization..."
            python tools/init_postgres_db.py
            echo "Database initialization completed."
        else
            echo "Database tables already exist (found $TABLE_COUNT tables). Skipping initialization."
        fi
        
        # Clean up temp file
        rm -f /tmp/db_check_output.txt
        
        cd ..
    else
        echo "Error: 'ai' directory not found"
        exit 1
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

# Create logs directory
mkdir -p ai/logs

# Start PostgreSQL and initialize DB (skip if using managed database)
if use_managed_db; then
    echo "Using managed PostgreSQL database (DATABASE_URL detected)"
    echo "Skipping local PostgreSQL setup"
    # Still run database initialization to create tables if needed
    init_database
else
    echo "Using local PostgreSQL database"
    start_postgres
    init_database
fi

# Start backend
echo "Starting Main..."
cd ai
python3 main.py &
cd ..

wait_for_service "http://0.0.0.0:8000/backend" "Backend"

# Confirm frontend
echo "Checking frontend..."
wait_for_service "http://0.0.0.0:8000" "Frontend"

# Start Qdrant
echo "Starting Qdrant..."
if check_qdrant; then
    qdrant --config-path <(echo '{"storage":{"storage_path":"/home/runner/workspace/qdrant_storage"},"service":{"host":"0.0.0.0","http_port":6333}}') &
    wait_for_service "http://0.0.0.0:6333/healthz" "Qdrant"
else
    echo "Continuing without Qdrant..."
fi

# Hold the script open
wait

echo "Services started. Check logs/ directory for output."
