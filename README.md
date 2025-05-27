# 🎵 Lo-Fi Song Generator

A powerful AI-powered application that generates Suno lo-fi prompts and YouTube-ready content using Google's Gemini AI. Create cohesive lo-fi albums with detailed prompts, YouTube metadata, and cover art suggestions.

## ✨ Features

- 🤖 AI-powered song prompt generation using Google's Gemini AI
- 🎼 Generate cohesive lo-fi albums with multiple tracks
- 📝 Detailed Suno AI prompts for each song
- 📺 YouTube-ready content including:
  - SEO-optimized titles
  - Detailed descriptions
  - Tags for discoverability
  - Cover art prompts
  - Thumbnail suggestions
- 💾 Export functionality:
  - CSV file for album/song data
  - Text file for YouTube content
- 🎨 Modern, user-friendly Gradio interface

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Gemini API key (get it from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lo-fi-mix
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the application:
```bash
python lofi_mix.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:7860)

## 🎮 How to Use

1. **Enter Your API Key**
   - Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Enter it in the password field

2. **Describe Your Style**
   - Enter a detailed description of the style, mood, or theme you want
   - Examples:
     - "Nostalgic Japanese summer evening with cicadas and gentle rain"
     - "Cozy winter study session with warm piano and crackling fireplace"
     - "Dreamy midnight cityscape with neon lights and soft jazz"

3. **Set Number of Songs**
   - Choose how many songs you want to generate (1-30)
   - Default is set to 3 songs

4. **Generate Content**
   - Click "Generate Lo-fi Album"
   - Wait for the AI to process your request

5. **Review and Download**
   - View the generated album content in the interface
   - Download the CSV file for song data
   - Download the text file for YouTube content

## 📝 Output Format

### Album CSV File
- Track number
- Song title
- Genre
- Suno AI prompt

### YouTube Content File
- Cover image prompts
- SEO-optimized titles
- Detailed descriptions
- Tags
- Thumbnail elements

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Google Gemini AI](https://deepmind.google/technologies/gemini/) for the AI capabilities
- [Gradio](https://gradio.app/) for the web interface
- [LangChain](https://www.langchain.com/) for the AI workflow
- [Suno AI](https://suno.ai/) for the music generation platform

## ⚠️ Disclaimer

This tool is designed to generate prompts for Suno AI. You'll need to use these prompts with Suno AI separately to create the actual music. The application does not generate music directly.
