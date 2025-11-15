"""
Perspective generator using LLM to provide opposing viewpoints.
"""

import os
import aiohttp
import asyncio
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PerspectiveGenerator:
    """
    Generate opposing perspectives and balanced viewpoints using Gemini LLM.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the perspective generator.
        
        Args:
            api_key: Google Gemini API key (optional, can use env variable)
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if self.api_key:
            self.model_name = "gemini-2.5-flash"
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
            self.enabled = True
        else:
            self.enabled = False
    
    async def _call_gemini_api(self, prompt: str) -> str:
        """
        Call Gemini API asynchronously.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Generated text response
        """
        headers = {
            'Content-Type': 'application/json'
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
                "responseMimeType": "text/plain"
            }
        }
        
        url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Debug: print response structure
                    print(f"API Response: {result}")
                    
                    # Handle different possible response structures
                    try:
                        # Standard structure
                        if 'candidates' in result and len(result['candidates']) > 0:
                            candidate = result['candidates'][0]
                            
                            # Check for parts in content
                            if 'content' in candidate and 'parts' in candidate['content']:
                                parts = candidate['content']['parts']
                                if len(parts) > 0 and 'text' in parts[0]:
                                    return parts[0]['text']
                            
                            # Check finish reason
                            finish_reason = candidate.get('finishReason', '')
                            if finish_reason == 'MAX_TOKENS':
                                raise Exception("Response was truncated due to token limit. Try shortening the article or reducing max_length.")
                            
                        raise Exception(f"No text content in response. Finish reason: {result.get('candidates', [{}])[0].get('finishReason', 'unknown')}")
                    except Exception as e:
                        if "Response was truncated" in str(e) or "No text content" in str(e):
                            raise
                        raise Exception(f"Error parsing API response: {str(e)}")
                else:
                    error_text = await response.text()
                    raise Exception(f"API call failed (status {response.status}): {error_text}")
    
    def generate_opposing_perspectives(
        self, 
        article_text: str, 
        detected_bias: str,
        max_length: int = 500
    ) -> Dict[str, any]:
        """
        Generate opposing perspectives to the detected bias.
        
        Args:
            article_text: The original article text
            detected_bias: The detected bias (Left, Centre, Right)
            max_length: Maximum length of generated perspectives
            
        Returns:
            Dictionary with opposing perspectives and balanced view
        """
        # Run async function synchronously
        return asyncio.run(self._generate_opposing_perspectives_async(
            article_text, detected_bias, max_length
        ))
    
    async def _generate_opposing_perspectives_async(
        self, 
        article_text: str, 
        detected_bias: str,
        max_length: int = 500
    ) -> Dict[str, any]:
        """
        Generate opposing perspectives to the detected bias.
        
        Args:
            article_text: The original article text
            detected_bias: The detected bias (Left, Centre, Right)
            max_length: Maximum length of generated perspectives
            
        Returns:
            Dictionary with opposing perspectives and balanced view
        """
        if not self.enabled:
            return {
                "success": False,
                "error": "LLM not configured. Please set GEMINI_API_KEY environment variable."
            }
        
        try:
            # Determine opposing perspectives to generate
            if detected_bias == "Left":
                opposing = ["Right", "Centre"]
                bias_description = "left-leaning/progressive"
            elif detected_bias == "Right":
                opposing = ["Left", "Centre"]
                bias_description = "right-leaning/conservative"
            else:  # Centre
                opposing = ["Left", "Right"]
                bias_description = "centrist/neutral"
            
            # Generate perspectives
            perspectives = {}
            
            # Create a summary first
            summary = await self._generate_summary(article_text)
            perspectives["summary"] = summary
            
            # Generate opposing viewpoints
            for perspective_type in opposing:
                perspective = await self._generate_perspective(
                    article_text, 
                    summary,
                    detected_bias, 
                    perspective_type,
                    max_length
                )
                perspectives[perspective_type.lower()] = perspective
            
            # Generate a balanced synthesis
            balanced = await self._generate_balanced_view(
                article_text,
                summary,
                detected_bias,
                perspectives,
                max_length
            )
            perspectives["balanced"] = balanced
            
            return {
                "success": True,
                "detected_bias": detected_bias,
                "perspectives": perspectives
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate perspectives: {str(e)}"
            }
    
    async def _generate_summary(self, article_text: str) -> str:
        """Generate a brief summary of the article."""
        prompt = f"""
Provide a brief, neutral 2-3 sentence summary of the main points in this article:

{article_text[:2000]}

Summary:"""
        
        response = await self._call_gemini_api(prompt)
        return response.strip()
    
    async def _generate_perspective(
        self, 
        article_text: str,
        summary: str,
        detected_bias: str,
        perspective_type: str,
        max_length: int
    ) -> str:
        """Generate a specific opposing perspective."""
        
        if perspective_type == "Left":
            stance = "progressive/left-leaning"
            focus = "social justice, equality, government intervention, collective welfare, and progressive reforms"
        elif perspective_type == "Right":
            stance = "conservative/right-leaning"
            focus = "individual liberty, free markets, limited government, traditional values, and fiscal responsibility"
        else:  # Centre
            stance = "centrist/moderate"
            focus = "balanced compromise, pragmatic solutions, bipartisan cooperation, and evidence-based policy"
        
        prompt = f"""
The following article was detected as having a {detected_bias} bias:

SUMMARY: {summary}

FULL ARTICLE (excerpt):
{article_text[:1500]}

Task: Write a {stance} perspective on this topic that presents an opposing or alternative viewpoint. 
Focus on: {focus}

Guidelines:
- Present 3-5 key counterarguments or alternative viewpoints
- Use respectful, factual language
- Highlight different priorities or values
- Keep it under {max_length} words
- Be constructive, not dismissive

{perspective_type} Perspective:"""
        
        response = await self._call_gemini_api(prompt)
        return response.strip()
    
    async def _generate_balanced_view(
        self,
        article_text: str,
        summary: str,
        detected_bias: str,
        perspectives: Dict[str, str],
        max_length: int
    ) -> str:
        """Generate a balanced synthesis of all perspectives."""
        
        prompt = f"""
Article Summary: {summary}
Detected Bias: {detected_bias}

Different perspectives have been presented on this topic. Now provide a balanced, nuanced view that:

1. Acknowledges valid points from multiple perspectives
2. Identifies common ground or shared concerns
3. Presents a more complete picture of the issue
4. Suggests constructive paths forward

Keep it under {max_length} words and maintain a neutral, thoughtful tone.

Balanced Perspective:"""
        
        response = await self._call_gemini_api(prompt)
        return response.strip()
    
    def generate_discussion_questions(
        self, 
        article_text: str, 
        detected_bias: str
    ) -> List[str]:
        """
        Generate thought-provoking discussion questions.
        
        Args:
            article_text: The original article
            detected_bias: Detected bias label
            
        Returns:
            List of discussion questions
        """
        # Run async function synchronously
        return asyncio.run(self._generate_discussion_questions_async(
            article_text, detected_bias
        ))
    
    async def _generate_discussion_questions_async(
        self, 
        article_text: str, 
        detected_bias: str
    ) -> List[str]:
        """
        Generate thought-provoking discussion questions.
        
        Args:
            article_text: The original article
            detected_bias: Detected bias label
            
        Returns:
            List of discussion questions
        """
        if not self.enabled:
            return []
        
        try:
            prompt = f"""
Based on this {detected_bias}-biased article, generate 5 thoughtful discussion questions that:
- Encourage critical thinking
- Explore different perspectives
- Challenge assumptions
- Promote nuanced understanding

Article excerpt:
{article_text[:1000]}

Provide exactly 5 questions, one per line, without numbering:"""
            
            response = await self._call_gemini_api(prompt)
            questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
            
            # Clean up any numbering that might have been added
            questions = [q.lstrip('0123456789.-) ') for q in questions]
            
            return questions[:5]
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []
