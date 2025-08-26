#!/usr/bin/env python3
"""
Gradio chat frontend for the Marin model API.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import gradio as gr
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_MODEL_NAME = "Marin Model"

class ChatMessage:
    """Represents a chat message."""
    
    def __init__(self, role: str, content: str, timestamp: Optional[float] = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }

class ChatUI:
    """Main chat UI class."""
    
    def __init__(self, api_url: str = DEFAULT_API_URL):
        self.api_url = api_url
        self.chat_history: List[ChatMessage] = []
        self.is_streaming = False
        
        # Initialize the Gradio interface
        self.interface = self._create_interface()
    
    def _create_interface(self) -> gr.Interface:
        """Create the Gradio interface."""
        
        with gr.Blocks(
            title="Marin Chat Interface",
            theme=gr.themes.Soft(),
            css="""
            .chat-container { height: 70vh; overflow-y: auto; }
            .message { margin: 10px 0; padding: 10px; border-radius: 10px; }
            .user-message { background-color: #e3f2fd; margin-left: 20%; }
            .assistant-message { background-color: #f3e5f5; margin-right: 20%; }
            .system-message { background-color: #fff3e0; text-align: center; }
            .streaming-indicator { 
                padding: 10px; 
                border-radius: 5px; 
                background-color: #e8f5e8; 
                border: 1px solid #4caf50;
                text-align: center;
                font-weight: bold;
            }
            """
        ) as interface:
            
            gr.Markdown("# 🤖 Marin Chat Interface")
            gr.Markdown("Chat with the Marin language model using the API server.")
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Chat display area
                    chat_display = gr.HTML(
                        value=self._format_chat_history(),
                        elem_classes=["chat-container"],
                        label="Chat History"
                    )
                    
                    # Input area
                    with gr.Row():
                        message_input = gr.Textbox(
                            placeholder="Type your message here...",
                            label="Message",
                            lines=3,
                            scale=4
                        )
                        send_button = gr.Button("Send", variant="primary", scale=1)
                    
                    # Clear chat button
                    clear_button = gr.Button("Clear Chat", variant="secondary")
                    
                    # Streaming indicator
                    streaming_indicator = gr.HTML(
                        value="",
                        label="Status",
                        elem_classes=["streaming-indicator"]
                    )
                
                with gr.Column(scale=1):
                    # Model status
                    status_box = gr.Textbox(
                        value="Checking API status...",
                        label="API Status",
                        interactive=False
                    )
                    
                    # Generation parameters
                    gr.Markdown("### Generation Parameters")
                    
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.5,
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness (0.0 = deterministic, 2.0 = very random)"
                    )
                    
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p",
                        info="Nucleus sampling parameter"
                    )
                    
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=20,
                        step=1,
                        label="Top-k",
                        info="Top-k sampling parameter"
                    )
                    
                    max_length = gr.Slider(
                        minimum=50,
                        maximum=4096,
                        value=80,
                        step=10,
                        label="Max Length",
                        info="Maximum number of tokens to generate"
                    )
                    
                    repetition_penalty = gr.Slider(
                        minimum=0.8,
                        maximum=2.0,
                        value=1.1,
                        step=0.1,
                        label="Repetition Penalty",
                        info="Penalty for repeating tokens"
                    )
                    
                    # Thinking tokens configuration
                    gr.Markdown("### Thinking Tokens")
                    enable_thinking = gr.Checkbox(
                        value=True,
                        label="Enable Thinking Tokens",
                        info="Allow the model to use thinking tokens before responding"
                    )
                    
                    max_think_effort = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=50,
                        step=1,
                        label="Max Thinking Tokens",
                        info="Maximum thinking effort",
                        visible=True
                    )
                    
                    # API configuration
                    gr.Markdown("### API Configuration")
                    api_url_input = gr.Textbox(
                        value=self.api_url,
                        label="API URL",
                        info="URL of the Marin model API server"
                    )
                    
                    test_connection_button = gr.Button("Test Connection", variant="secondary")
            
            # Event handlers
            send_button.click(
                fn=self._send_message,
                inputs=[
                    message_input,
                    temperature,
                    top_p,
                    top_k,
                    max_length,
                    repetition_penalty,
                    enable_thinking,
                    max_think_effort
                ],
                outputs=[chat_display, message_input, streaming_indicator],
                show_progress=True
            )
            
            message_input.submit(
                fn=self._send_message,
                inputs=[
                    message_input,
                    temperature,
                    top_p,
                    top_k,
                    max_length,
                    repetition_penalty,
                    enable_thinking,
                    max_think_effort
                ],
                outputs=[chat_display, message_input, streaming_indicator],
                show_progress=True
            )
            
            clear_button.click(
                fn=self._clear_chat,
                outputs=[chat_display]
            )
            
            test_connection_button.click(
                fn=self._test_connection,
                inputs=[api_url_input],
                outputs=[status_box]
            )
            
            # Thinking tokens toggle
            enable_thinking.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[enable_thinking],
                outputs=[max_think_effort]
            )
            
            # Update max thinking effort based on max length
            max_length.change(
                fn=lambda x: gr.update(maximum=min(x, 100)-10),
                inputs=[max_length],
                outputs=[max_think_effort]
            )
            
            # Initial status check
            interface.load(
                fn=self._check_api_status,
                outputs=[status_box]
            )
        
        return interface
    
    def _format_chat_history(self) -> str:
        """Format chat history as HTML."""
        if not self.chat_history:
            return '<div class="message system-message">No messages yet. Start a conversation!</div>'
        
        html_parts = []
        for message in self.chat_history:
            timestamp = time.strftime("%H:%M", time.localtime(message.timestamp))
            css_class = f"{message.role}-message"
            # Convert \n to HTML line breaks and style thinking tokens
            formatted_content = message.content.replace('\n', '<br>')
            
            # Style thinking tokens as light gray text
            if '<|start_think|>' in formatted_content:
                # Split content into parts: before thinking, thinking section, after thinking
                parts = formatted_content.split('<|start_think|>')
                if len(parts) > 1:
                    before_thinking = parts[0]
                    thinking_and_after = parts[1].split('<|end_think|>')
                    
                    if len(thinking_and_after) > 1:
                        # Both start and end tokens are present
                        thinking_section = thinking_and_after[0]
                        after_thinking = thinking_and_after[1]
                        
                        formatted_content = (
                            f'{before_thinking}'
                            f'<span style="color: #888888; font-style: italic; font-size: 0.9em;">'
                            f'<|start_think|>{thinking_section}<|end_think|>'
                            f'</span>'
                            f'{after_thinking}'
                        )
                    else:
                        # Only start token is present (during streaming)
                        thinking_section = thinking_and_after[0]
                        
                        formatted_content = (
                            f'{before_thinking}'
                            f'<span style="color: #888888; font-style: italic; font-size: 0.9em;">'
                            f'<|start_think|>{thinking_section}'
                            f'</span>'
                        )
            
            html_parts.append(
                f'<div class="message {css_class}">'
                f'<strong>{message.role.title()}</strong> ({timestamp})<br>'
                f'{formatted_content}'
                f'</div>'
            )
        
        return '\n'.join(html_parts)
    
    def _add_message(self, role: str, content: str):
        """Add a message to the chat history."""
        message = ChatMessage(role=role, content=content)
        self.chat_history.append(message)
    
    def _clear_chat(self) -> str:
        """Clear the chat history."""
        self.chat_history.clear()
        return self._format_chat_history()
    
    def _check_api_status(self) -> str:
        """Check the API server status."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return f"✅ API Online - Model: {data.get('model_name', 'Unknown')} - Device: {data.get('device', 'Unknown')}"
            else:
                return f"⚠️ API Error - Status: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return f"❌ API Offline - {str(e)}"
    
    def _test_connection(self, api_url: str) -> str:
        """Test connection to the API server."""
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return f"✅ Connected to {api_url}\nModel: {data.get('model_name', 'Unknown')}\nDevice: {data.get('device', 'Unknown')}"
            else:
                return f"⚠️ API Error - Status: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return f"❌ Connection Failed - {str(e)}"
    
    def _send_message(
        self,
        message: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_length: int,
        repetition_penalty: float,
        enable_thinking: bool,
        max_think_effort: int
    ):
        """Send a message to the API and get a response."""
        if not message.strip():
            return self._format_chat_history(), "", ""
        
        # Add user message to chat
        self._add_message("user", message.strip())
        
        try:
            # Prepare API request
            api_request = {
                "prompt": message.strip(),
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_length": max_length,
                "repetition_penalty": repetition_penalty,
                "stream": True,
                "do_sample": True,
                "num_return_sequences": 1,
                "max_think_effort": None,
            }
            
            # Add thinking tokens if enabled
            if enable_thinking:
                api_request["max_think_effort"] = max_think_effort
            
            # Initialize assistant message
            assistant_message = ""
            self._add_message("assistant", "")
            
            # Show streaming started
            yield self._format_chat_history(), "", "🔄 Generating..."
            
            # Use streaming endpoint for real-time updates
            response = requests.post(
                f"{self.api_url}/chat/stream",
                json=api_request,
                timeout=600,
                stream=True,
                headers={"Accept": "text/event-stream"}
            )
            
            if response.status_code == 200:
                # Process streaming response
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data_str = line[6:]  # Remove 'data: ' prefix
                            
                            if data_str == '[DONE]':
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    choice = data['choices'][0]
                                    if 'delta' in choice and 'content' in choice['delta']:
                                        content = choice['delta']['content']
                                        if content:
                                            assistant_message += content
                                            # Update the last message in chat history
                                            self.chat_history[-1].content = assistant_message
                                            # Yield the updated chat display for real-time updates
                                            yield self._format_chat_history(), "", "🔄 Generating..."
                            except json.JSONDecodeError:
                                continue
                
                # Clean up the final message
                if assistant_message:
                    assistant_message = assistant_message.replace("<|eot_id|>", "").strip()
                    self.chat_history[-1].content = assistant_message
                
                logger.info(f"Generated {len(assistant_message)} characters via streaming")
                
                # Show completion status
                yield self._format_chat_history(), "", "✅ Generation complete"
                
            else:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                self._add_message("system", error_msg)
                logger.error(error_msg)
                yield self._format_chat_history(), "", f"❌ Error: {response.status_code}"
        
        except requests.exceptions.RequestException as e:
            error_msg = f"Connection Error: {str(e)}"
            self._add_message("system", error_msg)
            logger.error(error_msg)
            yield self._format_chat_history(), "", f"❌ Connection Error"
        
        except Exception as e:
            error_msg = f"Unexpected Error: {str(e)}"
            self._add_message("system", error_msg)
            logger.error(error_msg)
            yield self._format_chat_history(), "", f"❌ Unexpected Error"
        
        # Return updated chat display, clear input, and reset streaming indicator
        return self._format_chat_history(), "", ""
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        return self.interface.launch(**kwargs)

def main():
    """Main function to run the chat UI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Marin Chat UI")
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"URL of the Marin model API server (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio interface on (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link for the interface"
    )
    
    args = parser.parse_args()
    
    # Create and launch the chat UI
    chat_ui = ChatUI(api_url=args.api_url)
    
    print(f"🚀 Starting Marin Chat UI...")
    print(f"📡 API Server: {args.api_url}")
    print(f"🌐 Web Interface: http://localhost:{args.port}")
    
    chat_ui.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
