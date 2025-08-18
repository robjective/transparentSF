# TransparentSF

A data analysis and visualization platform for San Francisco city data

## Overview

TransparentSF is a web-based application that combined AI agents with public data to answer questions, build interactive visualizations and deep analysis of San Francisco public city data. It includes features for:

- Automated analysis of public datasets
- AI-powered insights generation
- Interactive chat interface for data exploration

- Anomaly detection with PostgreSQL storage
- AI-powered newsletter narration with ElevenLabs text-to-speech

## Technology Stack

- **Backend**: Python
- **Vector Database**: Qdrant
- **Database**: PostgreSQL

- **APIs**:  
  - OpenAI API for analysis  

  - ElevenLabs API for text-to-speech narration

## Installation

**Prerequisites:**  
- Python3 with `pip`  
- Docker (for Qdrant) 
- PostgreSQL (for anomaly storage)
- OpenAI API key


**Steps:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/transparentSF.git
   cd transparentSF
   ```

2. **Install Python dependencies:**
   ```bash
   cd ai
   pip install -r requirements.txt
   cd ..
   ```

3. **Set up environment variables:**  
   Create a `.env` file in the project root with the following variables:
   ```env
   OPENAI_API_KEY=your_openai_api_key

   PERPLEXITY_API_KEY=your_perplexity_api_key # Optional: For automated newsletter generation
   ELEVENLABS_API_KEY=your_elevenlabs_api_key # Optional: For newsletter narration
   ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM # Optional: Default is Rachel voice
   DATAWRAPPER_API_KEY=your_datawrapper_api_key # Required: For map and chart generation
   
   # PostgreSQL Connection Details (Defaults are usually sufficient if running locally)
   PG_HOST=localhost 
   PG_PORT=5432
   PG_USER=postgres
   PG_PASSWORD=your_postgres_password # Set this if you configured a password
   PG_DBNAME=transparentsf 
   ```

4. **Set up PostgreSQL:**
   ```bash
   # For macOS with Homebrew
   brew install postgresql
   brew services start postgresql
   
   # For Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install postgresql postgresql-contrib
   
   # For Windows
   # Download and install from https://www.postgresql.org/download/windows/
   ```

5. **Initialize the PostgreSQL database:**
   ```bash
   python ai/tools/init_postgres_db.py
   ```
   
   This script will create:
   - A database named "transparentsf" (if it doesn't exist)
   - An "anomalies" table to store detected anomalies

   You can customize the connection parameters as needed:
   ```bash
   python ai/tools/init_postgres_db.py --host localhost --port 5432 --user postgres --password <pass> --dbname transparentsf
   ```
## Usage

1. **Start all services:**

   The easiest way to start all required services is to use the included script:
   ```bash
   ./start_services.sh
   ```
   
   This script will:
   - Start the main backend service
   - Start Qdrant for vector search

   - Ensure all services are ready before proceeding

2. **Access the application:**

   After starting the services, you can access:
   - **Chat Interface**: http://localhost:8000/
     - Use this for interactive data exploration and queries
   - **Backend Configuration**: http://localhost:8000/backend
     - Use this for system configuration, data analysis setup, and administration tasks
     - From here you can run analysis tasks, configure metrics, and manage the system

3. **Manual service startup (alternative):**

   If you prefer to start services individually:
   ```bash
   # Start Qdrant
   docker run -p 6333:6333 qdrant/qdrant
   
   # Start the backend service
   cd ai
   python main.py
   ```

4. **Run Initial Analysis:**
   
   To perform initial data analysis, visit the backend configuration at:
   ```
   http://localhost:8000/backend
   ```
   
   From the backend interface, you can:
   - Configure data sources
   - Run analysis on specific metrics
   - Schedule automatic analysis
   - View analysis results



## Project Structure

- `/ai`: Core analysis and processing scripts
  - `backend.py`: Initial data analysis pipeline
  - `webChat.py`: Interactive chat interface
  - `load_analysis_2_vec.py`: Vector database loader
  - `/agents`: AI agent system components
    - `explainer_agent.py`: Agent for explaining metric changes and anomalies
    - `__init__.py`: Agent package initialization
  - `/tools`: Utility scripts and tools
    - `anomaly_detection.py`: Anomaly detection with PostgreSQL storage
    - `init_postgres_db.py`: Database initialization tool
    - `view_anomalies.py`: Tool for viewing anomalies in the database
    - `gen_map_dw.py`: Datawrapper map generation
    - `genChart.py`: Chart generation utilities
    - `genAggregate.py`: Data aggregation tools
  - `/output`: Generated analysis results
    - `/narration`: Generated audio narrations of newsletters
  - `/logs`: System and analysis logs
    - `/evals`: Agent evaluation logs
  - `/templates`: Web interface templates
  - `/data`: Data storage and configuration
    - `/prompts`: AI prompt templates
    - `/datasets`: Dataset storage


### Agent System

The project uses a sophisticated agent system for data analysis and explanation:

#### Core Agents
- **Explainer Agent**: Specialized in explaining metric changes and anomalies
  - Handles metric analysis and visualization
  - Generates natural language explanations
  - Creates interactive charts and maps
  - Manages metric metadata and categorization

#### Agent Features
- **Tool Integration**:
  - Data querying and analysis
  - Chart and map generation
  - Anomaly detection
  - Metric management
  - Vector database search
- **Context Management**:
  - Maintains conversation history
  - Tracks analysis context
  - Manages dataset state
- **Evaluation System**:
  - Logs agent interactions
  - Tracks tool usage
  - Monitors performance metrics

#### Agent Tools
- **Data Tools**:
  - Dataset loading and management
  - Data aggregation and transformation
  - Time series analysis
- **Visualization Tools**:
  - Interactive map generation
  - Chart creation
  - Data presentation
- **Analysis Tools**:
  - Anomaly detection
  - Metric tracking
  - Trend analysis

For more details on agent configuration and usage, see the [Agent System Documentation](ai/README_AGENTS.md).

## Features

### Newsletter Narration

TransparentSF includes an AI-powered newsletter narration feature that converts monthly reports into professional-quality audio using ElevenLabs' text-to-speech API. The process involves:

1. **Audio Transformation**: AI optimization of newsletter content for audio consumption
   - Converts complex sentences into natural speech patterns
   - Writes out numbers and percentages for clear pronunciation
   - Removes visual elements (charts, links, emojis)
   - Adds smooth transitions between sections
   - Optimizes length for a 2-3 minute listening experience

2. **Text-to-Speech**: Generation of professional-quality audio narration using ElevenLabs API

To use the narration feature:
1. Generate a monthly newsletter as usual
2. Navigate to the Monthly Reports page
3. Find your desired report in the list
4. Click the "Generate Narration" button
5. Wait for processing (30-60 seconds)
6. The audio file will be saved in `output/narration/` with the same naming convention as the report

For more details on customization options, voice settings, and troubleshooting, see the [Newsletter Narration Documentation](ai/README_NARRATION.md).

### Interactive Maps and Charts

TransparentSF integrates with Datawrapper to create professional, interactive visualizations:

#### Map Types
- **Supervisor District Maps**: Visualize data across San Francisco's 11 supervisor districts
- **Police District Maps**: Show data distribution across SFPD districts
- **Point Maps**: Display specific locations with customizable markers
- **Address Maps**: Plot locations using addresses (automatic geocoding)
- **Intersection Maps**: Highlight specific street intersections

#### Map Features
- **Interactive Elements**:
  - Hover tooltips with detailed information
  - Zoom and pan controls
  - Responsive design for all devices
  - Custom color schemes and legends
- **Data Visualization**:
  - Support for both absolute values and percentage changes
  - Custom color palettes for different data types
  - Automatic scaling and normalization
  - Series grouping for multiple data categories

#### Chart Types
- **Time Series**: Track changes over time
- **Bar Charts**: Compare values across categories
- **Pie Charts**: Show proportional distributions
- **Line Charts**: Display trends and patterns

#### Customization Options
- **Visual Styling**:
  - Custom color schemes
  - Adjustable marker sizes
  - Configurable tooltips
  - Custom legends and labels
- **Data Presentation**:
  - Automatic data formatting
  - Custom number formatting
  - Configurable thresholds
  - Series grouping and aggregation

#### Usage
1. Generate a visualization through the web interface
2. Choose the appropriate map or chart type
3. Configure visualization settings
4. Preview and adjust as needed
5. Embed in reports or export as needed

For more details on customization options and advanced features, see the [Datawrapper Integration Documentation](ai/README_DATAWRAPPER.md).

## Contributing

1. Fork the repository  
2. Create your feature branch:  
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:  
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch:  
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

## License

This project is licensed under the ISC License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- San Francisco's DataSF efforts and all the departments that publish data
- OpenAI for AI capabilities


---

**Need help?** Feel free to open an issue for support.
