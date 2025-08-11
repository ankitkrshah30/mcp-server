import os
import asyncio
from dotenv import load_dotenv
from typing import Annotated

# Imports from the official starter kit
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
# --- FINAL FIX PART 1 ---
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
# --- FINAL FIX PART 2 ---
from mcp.types import Field, INTERNAL_ERROR

# Our imports for the custom tool
import google.generativeai as genai
import requests
import base64
import io
from PIL import Image

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


# --- Our Custom Tool: analyze_business_image ---
@mcp.tool(description="Analyzes an image of a local business and returns a JSON object with its details.")
async def analyze_business_image(
        puch_image_data: Annotated[str, Field(description="Base64-encoded image data to analyze.")]
) -> str:
    """Takes base64 image data, analyzes it with Gemini, and returns a JSON string."""
    print("🧠 Analyzing business image...")
    try:
        # Decode the base64 image from Puch
        image_bytes = base64.b64decode(puch_image_data)
        img = Image.open(io.BytesIO(image_bytes))

        # Prepare for Gemini
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        prompt = """You are an expert at analyzing images of local Indian businesses. Analyze the provided image and respond ONLY with a single JSON object that strictly follows this structure: { "businessType": "...", "tags": ["...", "..."], "description": "..." }. The description should be a friendly, one-sentence summary."""

        # Send to Gemini
        response = model.generate_content([prompt, img])

        # Clean the response to get pure JSON
        raw_text = response.text
        first_brace = raw_text.find('{')
        last_brace = raw_text.rfind('}')
        if first_brace != -1 and last_brace != -1:
            clean_json = raw_text[first_brace:last_brace + 1]
            print("✅ Analysis complete.")
            return clean_json
        else:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="Failed to generate valid JSON from AI response."))
    except Exception as e:
        print(f"❌ An error occurred during analysis: {e}")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))


# --- Run MCP Server (Following the Official Blueprint) ---
app = mcp.streamable_http_app()

if __name__ == "__main__":
    import uvicorn

    # The README says the default port is 8086
    uvicorn.run(app, host="0.0.0.0", port=8086)