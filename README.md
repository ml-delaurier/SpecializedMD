# SpecializedMD - Medical Education Platform

A comprehensive medical education platform focused on colorectal surgery techniques, combining multimedia content, expert annotations, and AI-powered learning assistance.

<div align="center">
  <img src="assets\logo.webp" alt="SpecializedMD Logo" width="800">
</div>

## Project Overview

SpecializedMD is designed to create an enriched learning environment by:
- Managing multimedia content (video, audio, transcripts)
- Supporting professor annotations and cut-away explanations
- Integrating external medical literature and research
- Generating synthetic Q&A data for enhanced learning
- Providing an AI-powered question answering system

## Project Structure

```
SpecializedMD/
├── api/                    # Backend API endpoints
├── assets/                # Project assets and media
├── core/                  # Core business logic
│   ├── content/          # Content management
│   ├── annotation/       # Annotation processing
│   ├── settings/         # Application settings
│   ├── rag/             # Retrieval-augmented generation
│   └── qa/              # Question-Answer generation
├── data/                 # Data storage
│   ├── lectures/         # Raw lecture content
│   ├── annotations/      # Professor annotations
│   └── external/         # External literature
├── models/               # AI model implementations
├── services/            # External service integrations
├── web/                 # Web interface
└── utils/               # Utility functions
```

## Features

1. Content Management
   - Video/audio lecture storage
   - Transcript generation and management
   - Annotation system for professor insights

2. Knowledge Integration
   - External medical literature integration
   - Case study management
   - Reference linking system

3. AI-Powered Learning
   - Question-Answer generation
   - Context-aware search
   - Learning path recommendations

4. Interactive Platform
   - Real-time annotation tools
   - Student feedback system
   - Settings management with secure API key storage

## Configuration

The platform uses a secure settings management system for handling API keys and configuration:

- API keys are stored securely in the user's home directory
- Automatic backup creation before settings changes
- User-friendly configuration UI
- Support for multiple external services:
  - PubMed/NCBI API for medical literature
  - AWS S3 for content storage
  - Groq and DeepSeek for AI models
  - UMLS for medical terminology

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure API keys using the settings UI:
   ```python
   from core.settings import show_settings
   show_settings()
   ```

## Contributing

We welcome contributions! Please see our contributing guidelines for more information.
