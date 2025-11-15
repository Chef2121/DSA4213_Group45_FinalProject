"""
Article extraction module for fetching and parsing articles from URLs.
"""

import requests
from newspaper import Article
from typing import Dict, Optional


class ArticleExtractor:
    """
    Extract article content from URLs using newspaper3k library.
    """
    
    def __init__(self):
        """Initialize the article extractor."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract_from_url(self, url: str) -> Dict[str, any]:
        """
        Extract article content from a URL.
        
        Args:
            url: URL of the article to extract
            
        Returns:
            Dictionary containing:
                - success: Whether extraction was successful
                - text: Extracted article text
                - title: Article title
                - authors: List of authors
                - publish_date: Publication date
                - error: Error message (if failed)
        """
        try:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                return {
                    "success": False,
                    "error": "Invalid URL. Must start with http:// or https://"
                }
            
            # Download and parse article
            article = Article(url)
            article.download()
            article.parse()
            
            # Check if we got content
            if not article.text or len(article.text.strip()) < 100:
                return {
                    "success": False,
                    "error": "Could not extract sufficient text from the article. The page may be behind a paywall or require login."
                }
            
            # Return extracted content
            return {
                "success": True,
                "text": article.text,
                "title": article.title or "Untitled",
                "authors": article.authors or [],
                "publish_date": article.publish_date.strftime("%Y-%m-%d") if article.publish_date else None,
                "url": url,
                "top_image": article.top_image or None
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Network error: Could not access URL. {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to extract article: {str(e)}"
            }
    
    def get_article_preview(self, text: str, max_words: int = 100) -> str:
        """
        Get a preview of the article text.
        
        Args:
            text: Full article text
            max_words: Maximum number of words in preview
            
        Returns:
            Preview text
        """
        words = text.split()
        if len(words) <= max_words:
            return text
        
        preview = " ".join(words[:max_words])
        return preview + "..."
