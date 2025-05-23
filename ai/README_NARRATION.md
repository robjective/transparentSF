# Newsletter Narration Feature

This feature allows you to generate audio narrations of monthly newsletters using ElevenLabs' text-to-speech API with AI-powered audio optimization.

## How It Works

The narration process involves two AI-powered steps:

1. **Audio Transformation**: The email-compatible newsletter is processed by an AI assistant to optimize it for audio consumption:
   - Converts complex sentences into natural speech patterns
   - Writes out numbers and percentages for clear pronunciation
   - Removes visual elements (charts, links, emojis)
   - Adds smooth transitions between sections
   - Optimizes length for a 2-3 minute listening experience

2. **Text-to-Speech**: The optimized script is then sent to ElevenLabs API to generate professional-quality audio narration.

## Setup

### 1. Get ElevenLabs API Key

1. Sign up for an account at [ElevenLabs](https://elevenlabs.io/)
2. Navigate to your API keys in the account settings
3. Generate a new API key

### 2. Configure Environment Variables

Add the following environment variables to your `.env` file:

```bash
# ElevenLabs Text-to-Speech API Configuration
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # Optional: Default is Rachel voice
```

### 3. Available Voices

You can use different voices by changing the `ELEVENLABS_VOICE_ID`. Popular options include:
- `21m00Tcm4TlvDq8ikWAM` - Rachel (default)
- `AZnzlk1XvdvUeBnXmlld` - Domi
- `29vD33N1CtxCmqQRPOHJ` - Drew
- `EXAVITQu4vr4xnSDxMaL` - Bella

You can find more voice IDs in your ElevenLabs dashboard.

## Usage

### Through the Web Interface

1. Generate a monthly newsletter as usual
2. Navigate to the Monthly Reports page
3. Find your desired report in the list
4. Click the "Generate Narration" button
5. Wait for processing (this may take 30-60 seconds)
6. The audio file will be saved in `output/narration/` with the same naming convention as the report

### File Naming Convention

Audio files follow the same naming pattern as reports:
- `monthly_report_0_2025_01_email.mp3` (for citywide January 2025 report)
- `monthly_report_3_2025_01_revised_email.mp3` (for District 3 January 2025 revised report)

## Audio Optimization Features

The AI transformation step automatically:

### Text Cleanup
- Removes HTML tags and formatting
- Eliminates chart references and visual elements
- Strips out hyperlinks while preserving important information

### Speech Optimization
- Converts "24%" to "twenty-four percent"
- Changes "SFPD" to "S-F-P-D" for clear pronunciation
- Writes "12.5%" as "twelve point five percent"

### Natural Flow
- Adds transition phrases between topics
- Creates conversational, podcast-style narration
- Maintains the newsletter's factual tone while making it audio-friendly

### Content Structure
- Provides clear introductions to each section
- Summarizes key points at the end
- Keeps optimal length for attention span (2-3 minutes)

## Troubleshooting

### Common Issues

**No audio generated**: Check that your ElevenLabs API key is valid and has sufficient credits.

**Audio cuts off**: The content may be too long. The system automatically truncates at sentence boundaries to stay within API limits.

**Poor pronunciation**: Try a different voice ID or consider customizing the audio transformation prompt.

### Error Messages

- `"No ElevenLabs API key found"`: Add `ELEVENLABS_API_KEY` to your `.env` file
- `"ElevenLabs API error: 401"`: Your API key is invalid or expired
- `"ElevenLabs API error: 429"`: You've exceeded your API rate limits

## Customization

### Voice Settings

You can modify voice settings in the code by adjusting:
```python
"voice_settings": {
    "stability": 0.5,      # 0.0-1.0 (lower = more variable)
    "similarity_boost": 0.5 # 0.0-1.0 (higher = more similar to training)
}
```

### Audio Transformation Prompt

The AI transformation prompt can be customized in `ai/data/prompts.json` under `monthly_report.audio_transformation` to adjust:
- Writing style and tone
- Length preferences
- Specific terminology handling
- Transition phrases

## File Structure

Narrated audio files are saved in:
```
ai/output/narration/
```

Files follow the same naming convention as newsletters:
- `monthly_report_0_2024_04_revised.mp3`

## Technical Details

- **Text Cleaning**: HTML tags, URLs, and formatting are automatically removed
- **Character Limit**: Text is limited to 2500 characters to stay within ElevenLabs API limits
- **Audio Format**: MP3 format with standard ElevenLabs quality settings
- **Voice Settings**: 
  - Stability: 0.5
  - Similarity Boost: 0.5

## API Limits

ElevenLabs has usage limits based on your subscription:
- Free tier: Limited characters per month
- Paid tiers: Higher limits and better quality

Check your usage at [ElevenLabs Usage Dashboard](https://elevenlabs.io/usage). 