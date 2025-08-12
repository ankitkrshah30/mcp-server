import os
import asyncio
from dotenv import load_dotenv
from typing import Annotated
import json  # We need this to parse the JSON from Gemini

# Imports from the official starter kit
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import Field, INTERNAL_ERROR

# Our imports for the custom tool
import google.generativeai as genai
import requests
import base64
import io
from PIL import Image
from pydantic import BaseModel  # Import BaseModel for our rich description

# --- Load environment variables with CORRECT names ---
load_dotenv()
TOKEN = os.getenv("AUTH_TOKEN")
MY_NUMBER = os.getenv("MY_NUMBER")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert GEMINI_API_KEY is not None, "Please set GEMINI_API_KEY in your .env file"

# --- Configure Gemini AI ---
genai.configure(api_key=GEMINI_API_KEY)


# --- Official Authentication Provider Class ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            print("✅ Auth token matched!")
            return AccessToken(token=token, client_id="puch-client", scopes=["*"], expires_at=None)
        print("❌ Auth token did NOT match!")
        return None


# --- MCP Server Setup (Following the Official Blueprint) ---
mcp = FastMCP(
    "Aas Paas AI Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)


# --- Tool: validate (Required by Puch) ---
@mcp.tool
async def validate() -> str:
    """This tool is called by Puch to verify the server owner."""
    print(f"✅ Validate tool called. Returning phone number: {MY_NUMBER}")
    return MY_NUMBER


# --- A RICHER TOOL DESCRIPTION ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str


AasPaasDescription = RichToolDescription(
    description="Analyzes an image of a local, unorganized Indian business (like a street vendor or small shop) and extracts structured information about it.",
    use_when="Use this tool when a user provides a photo of a small shop, stall, or local service in India and wants to know what it is."
)


# --- Our Custom Tool: analyze_business_image ---
@mcp.tool(description=AasPaasDescription.model_dump_json())
async def analyze_business_image(
        puch_image_data: Annotated[str, Field(description="Base64-encoded image data to analyze.")]
) -> str:
    """Takes base64 image data, analyzes it with the AI, and returns a formatted string."""
    print("🧠 Analyzing business image...")
    try:
        image_bytes = base64.b64decode(puch_image_data)
        img = Image.open(io.BytesIO(image_bytes))

        # A MORE POWERFUL PROMPT
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = """You are an expert analyst specializing in India's informal economy. Your task is to analyze the provided image of a local, unorganized business and return a structured JSON object.

        The JSON object must strictly follow this structure:
        {
          "businessName": "A creative, descriptive name for the business (e.g., 'Sharma Ji's Chai Stall')",
          "businessType": "A clear category (e.g., 'Street Food Vendor', 'Tailor Shop', 'Cobbler', 'Electronics Repair')",
          "estimatedLocation": "A general description of the location (e.g., 'Roadside stall in a busy market', 'Small shop in a residential alley')",
          "keyItems": ["A list of key items or services visible (e.g., 'Tea', 'Samosa', 'Sewing Machine', 'Shoes')", "...", "..."],
          "description": "A friendly, one-sentence summary for a user."
        }

        Analyze the image and provide ONLY the JSON object. Do NOT include any other text or markdown formatting."""

        response = model.generate_content([prompt, img])

        raw_text = response.text
        first_brace = raw_text.find('{')
        last_brace = raw_text.rfind('}')
        if first_brace != -1 and last_brace != -1:
            clean_json = raw_text[first_brace:last_brace + 1]

            # A FRIENDLIER OUTPUT FOR THE USER
            data = json.loads(clean_json)

            # --- THE CHANGE IS HERE ---
            # I've made the final line more generic.
            response_to_user = (
                f'I think I see a *{data.get("businessType", "Local Business")}*!\n\n'
                f'🏷️ **Name:** {data.get("businessName", "N/A")}\n'
                f'📍 **Location:** {data.get("estimatedLocation", "N/A")}\n\n'
                f'**What I can see:**\n'
            )
            for item in data.get("keyItems", []):
                response_to_user += f"- {item}\n"

            response_to_user += f'\n> {data.get("description", "No summary available.")}\n\n'
            response_to_user += f'*Powered by AI • For local businesses*'

            print("✅ Analysis complete. Returning formatted Markdown.")
            return response_to_user
        else:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="Failed to generate valid JSON from AI response."))
    except Exception as e:
        print(f"❌ An error occurred during analysis: {e}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))


# --- Run MCP Server ---
app = mcp.streamable_http_app()

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8086))
    uvicorn.run(app, host="0.0.0.0", port=port)