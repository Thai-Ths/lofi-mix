from typing import List, Optional, Any, Dict, TypedDict, Annotated
from pydantic import BaseModel, Field
import pandas as pd
import csv
import io
import os
# from dataclasses import dataclass
# from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import gradio as gr

# Enhanced State Models
class StyleData(BaseModel):
    """Enhanced style data with additional context for better song generation"""
    genre: List[str] = Field(description="Primary musical genres (e.g. lo-fi hip hop, jazzhop, chillhop)")
    mood: List[str] = Field(description="Emotional tones (e.g. nostalgic, dreamy, melancholic, peaceful)")
    instruments: List[str] = Field(description="Key instruments (e.g. piano, vinyl crackle, guitar, synth, drums)")
    concept: str = Field(description="Artistic or thematic concept summary")
    bpm_range: str = Field(description="Tempo range (e.g. 65-80 BPM)")
    audio_texture: str = Field(description="Sound characteristics (e.g. warm, grainy, cassette-like)")
    culture_or_season: Optional[str] = Field(description="Cultural reference or seasonal theme")
    
    # New fields for better context
    time_of_day: Optional[str] = Field(description="Preferred listening time (e.g. late night, morning, sunset)")
    activity_context: Optional[str] = Field(description="Activity context (e.g. studying, relaxing, working)")
    reference_artists: List[str] = Field(default=[], description="Similar artists or style references")

class SunoPrompt(BaseModel):
    """Enhanced Suno prompt with better structure"""
    song_name: str = Field(description="Catchy, descriptive title reflecting theme/vibe")
    genre: str = Field(description="Specific genre/sub-genre")
    song_prompt: str = Field(description="Detailed prompt for SunoAI including style, mood, instruments, atmosphere")
    # tags: List[str] = Field(description="Relevant tags for discoverability")
    # duration_hint: Optional[str] = Field(description="Suggested duration (e.g. '2-3 minutes')")

class Album(BaseModel):
    """Collection of songs forming a cohesive album"""
    album_name: str = Field(description="Overall album/playlist name")
    theme: str = Field(description="Unifying theme across all tracks")
    songs: List[SunoPrompt] = Field(description="List of song prompts")
    track_count: int = Field(description="Total number of tracks")

class YouTubeContent(BaseModel):
    """YouTube-optimized content for each song"""
    cover_image_prompt: str = Field(description="AI image generation prompt for cover art")
    title: str = Field(description="YouTube-optimized title with keywords")
    description: str = Field(description="Detailed description with tags and timestamps")
    tags: List[str] = Field(description="YouTube tags for discoverability")
    thumbnail_elements: List[str] = Field(description="Key visual elements for thumbnail")

class State(TypedDict):
    """Enhanced state with better tracking and error handling"""
    user_input: str
    processing_stage: str  # Track current processing stage
    style_data: Optional[StyleData]
    album: Optional[Album]
    number_of_song: int
    youtube_content: List[YouTubeContent]
    messages: Annotated[List[Any], add_messages]
    errors: List[str]  # Track any errors
    metadata: Dict[str, Any]  # Additional metadata

# Agent Implementation
class LoFiSongGenerator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.7)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("extract_style", self.extract_style_agent)
        workflow.add_node("generate_songs", self.song_write_agent)
        workflow.add_node("create_youtube_content", self.cover_title_agent)
        workflow.add_node("compile_output", self.compile_output)
        
        # Add edges
        workflow.add_edge(START, "extract_style")
        workflow.add_edge("extract_style", "generate_songs")
        workflow.add_edge("generate_songs", "create_youtube_content")
        workflow.add_edge("create_youtube_content", "compile_output")
        workflow.add_edge("compile_output", END)
        
        return workflow.compile()
    
    def extract_style_agent(self, state: State) -> State:
        """Agent 1: Extract style information from user input"""
        try:
            state["processing_stage"] = "Extracting style information..."
            
            system_prompt = """You are a music style analysis expert specializing in lo-fi and chill music genres.
            Analyze the user's input and extract detailed style information that will guide song generation.
            
            You MUST respond with a valid JSON object containing the following fields:
            {
                "genre": ["list of genres like lo-fi hip hop, jazzhop, chillhop"],
                "mood": ["list of moods like nostalgic, dreamy, melancholic"],
                "instruments": ["list of instruments like piano, vinyl, guitar, synth"],
                "concept": "short description of the artistic concept",
                "bpm_range": "tempo range like 65-80 BPM",
                "audio_texture": "sound characteristics like warm, grainy, cassette-like",
                "culture_or_season": "cultural or seasonal theme (optional)",
                "time_of_day": "preferred listening time (optional)",
                "activity_context": "activity context like studying, relaxing (optional)",
                "reference_artists": ["list of similar artists or references"]
            }
            
            Analyze the user's request and extract all relevant musical style information.
            Be creative and detailed in your analysis but ensure valid JSON format."""
            
            user_message = f"Analyze this music style request and return JSON: {state['user_input']}"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse the actual LLM response
            style_data = self._parse_style_response(response.content)
            
            state["style_data"] = style_data
            state["messages"].append(AIMessage(content=f"Style analysis complete: {style_data.concept}"))
            
        except Exception as e:
            state["errors"].append(f"Style extraction error: {str(e)}")
            
        return state
    
    def song_write_agent(self, state: State) -> State:
        """Agent 2: Generate song prompts based on style data"""
        try:
            state["processing_stage"] = "Generating song prompts..."
            
            if not state.get("style_data"):
                raise ValueError("No style data available")
            
            style_data = state["style_data"]
            number_of_song = state["number_of_song"]
            
            system_prompt = f"""You are a creative music producer specializing in lo-fi beats and chill music.
            Generate {number_of_song} unique song prompts that form a cohesive album based on the provided style data.
            
            You MUST respond with a valid JSON object in this format:
            {{
                "album_name": "Name of the album/playlist",
                "theme": "Unifying theme description",
                "track_count": 5,
                "songs": [
                    {{
                        "song_name": "Title of the song",
                        "genre": "Specific genre",
                        "song_prompt": "Detailed Suno AI prompt with instruments, mood, atmosphere"
                    }}
                ]
            }}
            
            Each song should:
            - Have a unique but related theme
            - Songs must be ordered by semantic proximity to the input style or mood.
            - Describe the mood and atmosphere clearly
            - Avoid repetition
            - Be optimized for SunoAI generation
            - Vary slightly in tempo and energy while maintaining cohesion
            
            """
            
            style_summary = f"""
            Genre: {', '.join(style_data.genre)}
            Mood: {', '.join(style_data.mood)}
            Instruments: {', '.join(style_data.instruments)}
            Concept: {style_data.concept}
            BPM: {style_data.bpm_range}
            Texture: {style_data.audio_texture}
            Context: {style_data.culture_or_season or 'General'}
            Time: {style_data.time_of_day or 'Anytime'}
            Activity: {style_data.activity_context or 'General listening'}
            """
            
            user_message = f"Generate a cohesive lo-fi album JSON based on this style analysis:\n{style_summary}"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse the actual LLM response
            album = self._parse_album_response(response.content)
            
            state["album"] = album
            state["messages"].append(AIMessage(content=f"Generated {len(album.songs)} songs for album: {album.album_name}"))
            
        except Exception as e:
            state["errors"].append(f"Song generation error: {str(e)}")
            
        return state
    
    def cover_title_agent(self, state: State) -> State:
        """Agent 3: Generate YouTube titles, descriptions, and cover art prompts"""
        try:
            state["processing_stage"] = "Creating YouTube content..."
            
            if not state.get("album"):
                raise ValueError("No album data available")
            
            album = state["album"]
            
            system_prompt = """You are a YouTube content optimization expert specializing in music content.
            For the provided album, create YouTube content for that album.
            
            You MUST respond with a valid JSON array in this format:
            [
                {
                    "cover_image_prompt": "Detailed AI image generation prompt for cover art",
                    "title": "SEO-optimized YouTube title with keywords",
                    "description": "Detailed description with hashtags and context",
                    "tags": ["youtube", "tags", "for", "discoverability"],
                    "thumbnail_elements": ["key", "visual", "elements"]
                }
            ]
            
            You should:
            1. SEO-optimized title that includes relevant keywords
            2. Detailed description with hashtags and timestamps
            3. AI image generation prompt for cover art
            4. List of YouTube tags for discoverability
            5. Thumbnail design elements
            6. Think 3 prompts difference
            
            Focus on lo-fi, chill, study music, and relaxation keywords.
            Make content discoverable but authentic."""
            
            album_info = f"""
            Album: {album.album_name}
            Theme: {album.theme}
            Songs: {[{'name': song.song_name, 'genre': song.genre} for song in album.songs]}
            """
            
            user_message = f"Create YouTube content JSON array for this album:\n{album_info}"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse the actual LLM response
            youtube_content = self._parse_youtube_response(response.content)
            
            state["youtube_content"] = youtube_content
            state["messages"].append(AIMessage(content=f"Created YouTube content for {len(youtube_content)} songs"))
            
        except Exception as e:
            state["errors"].append(f"YouTube content generation error: {str(e)}")
            
        return state
    
    def compile_output(self, state: State) -> State:
        """Compile final output into two separate CSV files"""
        try:
            state["processing_stage"] = "Compiling final output..."
            
            if not state.get("album") or not state.get("youtube_content"):
                raise ValueError("Missing required data for compilation")
            
            album = state["album"]
            youtube_content = state["youtube_content"]
            
            # Create Album/Song CSV data
            album_data = []
            for i, song in enumerate(album.songs):
                album_data.append({
                    "track_number": i + 1,
                    "song_name": song.song_name,
                    "genre": song.genre,
                    "suno_prompt": song.song_prompt
                })
            
            # Create YouTube CSV data
            youtube_text_output = ""
            for i, yt in enumerate(youtube_content, 1):
                youtube_text_output += f"ID {i}:\n"
                youtube_text_output += f"Title: {yt.title}\n"
                youtube_text_output += f"Description: {yt.description}\n"
                youtube_text_output += f"Tags: {', '.join(yt.tags)}\n"
                youtube_text_output += f"Cover Image Prompt: {yt.cover_image_prompt}\n"
                youtube_text_output += f"Thumbnail Elements: {', '.join(yt.thumbnail_elements)}\n"
                youtube_text_output += "\n"
                youtube_text_output += "-----------------------------------------------------------"
                youtube_text_output += "\n"
            
            # Get the directory where the app is running
            app_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Sanitize album name for filename
            import re
            sanitized_album_name = re.sub(r'[<>:"/\\|?*]', '_', album.album_name)
            sanitized_album_name = sanitized_album_name.strip()
            
            # Define file paths with proper extension
            album_filename = f"{sanitized_album_name}_songs.csv"
            youtube_filename = f"{sanitized_album_name}_youtube.txt"
            
            album_filepath = os.path.join(app_dir, album_filename)
            youtube_filepath = os.path.join(app_dir, youtube_filename)
            
            # Create StringIO objects for CSV content
            album_csv_content = io.StringIO()
            
            # Write Album CSV content to StringIO
            if album_data:
                album_fieldnames = album_data[0].keys()
                album_writer = csv.DictWriter(album_csv_content, fieldnames=album_fieldnames)
                album_writer.writeheader()
                album_writer.writerows(album_data)
                
                # Write to file
                with open(album_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    csvfile.write(album_csv_content.getvalue())
            
            # Write YouTube text output to file
            if youtube_text_output:
                with open(youtube_filepath, 'w', encoding='utf-8') as txtfile:
                    txtfile.write(youtube_text_output)
            
            # Store both CSV outputs and file paths in state
            state["metadata"]["album_csv"] = album_csv_content.getvalue()
            state["metadata"]["youtube_txt"] = youtube_text_output
            state["metadata"]["album_csv_file"] = album_filepath
            state["metadata"]["youtube_txt_file"] = youtube_filepath
            state["metadata"]["total_songs"] = len(album_data)
            state["metadata"]["album_name"] = album.album_name
            state["processing_stage"] = "Complete!"
            
        except Exception as e:
            state["errors"].append(f"Compilation error: {str(e)}")
            # Clear file paths if there was an error
            state["metadata"]["album_csv_file"] = None
            state["metadata"]["youtube_txt_file"] = None
        
        return state
    
    # Helper methods for parsing LLM responses
    def _parse_style_response(self, response_content: str) -> StyleData:
        """Parse LLM JSON response into StyleData object"""
        import json
        import re
        
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                return StyleData(
                    genre=data.get("genre", []),
                    mood=data.get("mood", []),
                    instruments=data.get("instruments", []),
                    concept=data.get("concept", "Lo-fi music"),
                    bpm_range=data.get("bpm_range"),
                    audio_texture=data.get("audio_texture"),
                    culture_or_season=data.get("culture_or_season"),
                    time_of_day=data.get("time_of_day"),
                    activity_context=data.get("activity_context"),
                    reference_artists=data.get("reference_artists", [])
                )
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract information from natural language
            return self._fallback_parse_style(response_content)
    
    def _fallback_parse_style(self, content: str) -> StyleData:
        """Fallback parser when JSON parsing fails"""
        # Simple keyword extraction as fallback
        content_lower = content.lower()
        
        genres = []
        if "lo-fi" in content_lower or "lofi" in content_lower:
            genres.append("lo-fi hip hop")
        if "jazz" in content_lower:
            genres.append("jazzhop")
        if "chill" in content_lower:
            genres.append("chillhop")
        if not genres:
            genres = ["lo-fi hip hop"]
            
        moods = []
        for mood in ["nostalgic", "dreamy", "peaceful", "melancholic", "relaxing", "cozy"]:
            if mood in content_lower:
                moods.append(mood)
        if not moods:
            moods = ["relaxing"]
            
        instruments = []
        for instrument in ["piano", "guitar", "vinyl", "drums", "synth", "saxophone"]:
            if instrument in content_lower:
                instruments.append(instrument)
        if not instruments:
            instruments = ["piano", "soft drums"]
            
        return StyleData(
            genre=genres,
            mood=moods,
            instruments=instruments,
            concept=f"Musical style based on: {content[:100]}...",
            bpm_range="slow beat BPM",
            audio_texture="warm and atmospheric",
            culture_or_season=None,
            time_of_day=None,
            activity_context=None,
            reference_artists=[]
        )
    
    def _parse_album_response(self, response_content: str) -> Album:
        """Parse LLM JSON response into Album object"""
        import json
        import re
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                songs = []
                for song_data in data.get("songs", []):
                    song = SunoPrompt(
                        song_name=song_data.get("song_name", "Untitled"),
                        genre=song_data.get("genre"),
                        song_prompt=song_data.get("song_prompt"),
                        # tags=song_data.get("tags", ["lofi", "chill"]),
                        # duration_hint=song_data.get("duration_hint", "2-3 minutes")
                    )
                    songs.append(song)
                
                return Album(
                    album_name=data.get("album_name", "Lo-Fi Collection"),
                    theme=data.get("theme", "Relaxing music"),
                    songs=songs,
                    track_count=len(songs)
                )
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError):
            # Fallback: create a basic album
            return self._fallback_parse_album(response_content)
    
    def _fallback_parse_album(self, content: str) -> Album:
        """Fallback parser when JSON parsing fails"""
        # Create basic songs from content
        songs = [
            SunoPrompt(
                song_name=f"Lo-Fi Track {i+1}",
                genre="lo-fi hip hop",
                song_prompt=f"A gentle lo-fi track with piano and soft drums, {content[:50]}...",
                # tags=["lofi", "chill", "relax"],
                # duration_hint="2-3 minutes"
            )
            for i in range(5)
        ]
        
        return Album(
            album_name="Generated Lo-Fi Album",
            theme="Relaxing lo-fi music collection",
            songs=songs,
            track_count=5
        )
    
    def _parse_youtube_response(self, response_content: str) -> List[YouTubeContent]:
        """Parse LLM JSON response into list of YouTubeContent objects"""
        import json
        import re
        
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                youtube_content = []
                for item in data:
                    yt_content = YouTubeContent(
                        cover_image_prompt=item.get("cover_image_prompt", "Lo-fi aesthetic artwork"),
                        title=item.get("title", "Lo-Fi Music"),
                        description=item.get("description", "Relaxing lo-fi beats"),
                        tags=item.get("tags", ["lofi", "chill"]),
                        thumbnail_elements=item.get("thumbnail_elements", ["music", "aesthetic"])
                    )
                    youtube_content.append(yt_content)
                
                return youtube_content
            else:
                raise ValueError("No JSON array found in response")
                
        except (json.JSONDecodeError, ValueError):
            # Fallback: return basic YouTube content
            return self._fallback_parse_youtube(response_content)
    
    def _fallback_parse_youtube(self, content: str) -> List[YouTubeContent]:
        """Fallback parser when JSON parsing fails"""
        # Create basic YouTube content
        return [
            YouTubeContent(
                cover_image_prompt="Aesthetic lo-fi scene with warm lighting and vintage elements",
                title="Lo-Fi Hip Hop Beats | Study & Relaxation Music",
                description="🎵 Relaxing lo-fi hip hop beats perfect for studying, working, or chilling out.\n\n#lofi #chillhop #studymusic #relaxation",
                tags=["lofi", "lo-fi hip hop", "chill beats", "study music", "relaxation"],
                thumbnail_elements=["warm colors", "vintage aesthetic", "musical notes"]
            )
            for _ in range(5)  # Default 5 songs
        ]
    
    def process(self, user_input: str, number_of_song: int) -> State:
        """Main processing function"""
        initial_state = State(
            user_input=user_input,
            processing_stage="Starting...",
            style_data=None,
            album=None,
            number_of_song=number_of_song,
            youtube_content=[],
            messages=[],
            errors=[],
            metadata={}
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        return final_state

# Gradio Interface
def create_gradio_interface():
    generator = LoFiSongGenerator()
    
    def process_request(user_input, number_of_song=5):
        if not user_input.strip():
            return "", "", "Please enter a style or mood description."
        
        try:
            final_state = generator.process(user_input, number_of_song)
            
            # Get CSV outputs from metadata
            album_csv = final_state.get("metadata", {}).get("album_csv", "No album data generated")
            youtube_txt = final_state.get("metadata", {}).get("youtube_txt", "No YouTube data generated")
            
            # Get additional info for status
            album_name = final_state.get("metadata", {}).get("album_name", "Unknown")
            total_songs = final_state.get("metadata", {}).get("total_songs", 0)
            status = final_state.get("processing_stage", "Unknown")

            # Get file paths
            album_csv_file = final_state.get("metadata", {}).get('album_csv_file')
            youtube_txt_file = final_state.get("metadata", {}).get('youtube_txt_file')
            
            # Format status with more details
            detailed_status = f"Status: {status}\n"
            detailed_status += f"Album: {album_name}\n"
            detailed_status += f"Total Songs Generated: {total_songs}\n"
            
            # Add file paths if available
            if album_csv_file:
                detailed_status += f"Album CSV saved to: {album_csv_file}\n"
            if youtube_txt_file:
                detailed_status += f"YouTube CSV saved to: {youtube_txt_file}\n"
            
            # Add any errors
            if final_state.get("errors"):
                detailed_status += f"\nErrors encountered: {', '.join(final_state['errors'])}"
            
            # Convert CSV string to DataFrame for Gradio Dataframe component
            import pandas as pd
            import io
            
            # Parse the CSV string into a DataFrame
            df = pd.read_csv(io.StringIO(album_csv))
            
            # Return the DataFrame and other outputs
            return df, youtube_txt, detailed_status, album_csv_file, youtube_txt_file
            
        except Exception as e:
            return pd.DataFrame(), "", f"Error: {str(e)}", None, None
    
    # Create Gradio interface
    def make_interface():
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🎵 Lo-Fi Song Generator")
            gr.Markdown("Generate lo-fi songs with YouTube-ready content using AI agents!")

            with gr.Column():
                user_input = gr.Textbox(
                    label="Style or Mood Input",
                    placeholder="Enter the style, mood, or theme you want (e.g., 'nostalgic Japanese summer evening', 'cozy winter study session', 'dreamy midnight vibes')",
                    lines=3
                )
                num_songs = gr.Slider(minimum=1, maximum=30, value=25, step=1, label="Number of Songs")

            generate_btn = gr.Button("Generate Lo-fi Album")

            gr.Markdown("### 🌟 Examples")
            gr.Examples(
                examples=[
                    ["Nostalgic Japanese summer evening with cicadas and gentle rain"],
                    ["Cozy winter study session with warm piano and crackling fireplace"],
                    ["Dreamy midnight cityscape with neon lights and soft jazz"],
                    ["Autumn morning coffee shop vibes with acoustic guitar"],
                    ["Peaceful forest walk with birds chirping and gentle breeze"]
                ],
                inputs=user_input
            )

            gr.Markdown("### 🎼 Album Content")
            album_output = gr.Dataframe(
                label="Album/Songs Output",
                headers=["Track", "Title", "Genre", "Suno AI Prompt"],
                row_count=15,
                col_count=(4, "fixed"),
                interactive=False,
                wrap=True
            )

            gr.Markdown("### 📺 YouTube Metadata")
            youtube_output = gr.Textbox(
                label="YouTube Content Output",
                lines=15,
                max_lines=25,
                info="Contains YouTube titles, descriptions, tags, and cover art prompts"
            )

            gr.Markdown("### ⚙️ Status & Logs")
            status_output = gr.Textbox(
                label="Processing Status & Details",
                lines=5,
                max_lines=15,
                info="Shows processing status, file paths, and any errors"
            )

            with gr.Row():
                csv_output = gr.File(
                    label="Download Album CSV",
                    file_types=[".csv"],
                    type="filepath",
                    interactive=False
                )
                txt_output = gr.File(
                    label="Download YouTube TXT",
                    file_types=[".txt"],
                    type="filepath",
                    interactive=False
                )

            generate_btn.click(
                fn=process_request,
                inputs=[user_input, num_songs],
                outputs=[album_output, youtube_output, status_output, csv_output, txt_output]
            )

        return demo

    interface = make_interface()
    
    return interface

# Main execution
if __name__ == "__main__":
    # Create and launch the Gradio interface
    interface = create_gradio_interface()
    interface.launch()