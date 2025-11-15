"""
Article fetcher module for finding related articles using NewsAPI and optional Gemini AI.
"""

import os
from datetime import datetime
from typing import List, Dict, Optional
from newspaper import Article
from newsapi import NewsApiClient

try:
    from google import genai
    from pydantic import BaseModel
    
    class NewsQuery(BaseModel):
        """Schema for Gemini AI response."""
        query: str
        from_param: Optional[str] = None
        to: Optional[str] = None
    
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    NewsQuery = None
    GENAI_AVAILABLE = False


class ArticleFetcher:
    """Fetches related articles based on a seed article URL."""
    
    def __init__(self, newsapi_key: Optional[str] = None, gemini_key: Optional[str] = None):
        """
        Initialize the article fetcher.
        
        Args:
            newsapi_key: NewsAPI key (defaults to env var NEWSAPI_KEY)
            gemini_key: Google Gemini API key (defaults to env var GEMINI_API_KEY)
        """
        self.newsapi_key = newsapi_key or os.getenv("NEWSAPI_KEY")
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")
        
        if not self.newsapi_key:
            raise ValueError("NewsAPI key not provided. Set NEWSAPI_KEY environment variable.")
        
        self.api = NewsApiClient(api_key=self.newsapi_key)
        
        # Initialize Gemini client if available and key provided
        self.genai_client = None
        if GENAI_AVAILABLE and self.gemini_key:
            try:
                self.genai_client = genai.Client(api_key=self.gemini_key)
            except Exception as e:
                print(f"Warning: Could not initialize Gemini client: {e}")
    
    def extract_article(self, url: str) -> Dict:
        """
        Extract article content and metadata from URL.
        
        Args:
            url: Article URL
            
        Returns:
            Dictionary with article metadata and text
        """
        article = Article(url)
        article.download()
        article.parse()
        
        metadata = article.meta_data or {}
        
        return {
            "url": url,
            "title": article.title or "",
            "text": article.text or "",
            "description": metadata.get("description", ""),
            "tags": metadata.get("tags", []),
            "publishedAt": metadata.get("article", {}).get("published_time") or 
                          metadata.get("og", {}).get("published_time"),
            "source": None,
        }
    
    def generate_search_query(self, article_meta: Dict) -> str:
        """
        Generate an unbiased search query from article metadata.
        Uses Gemini AI if available, otherwise falls back to simple extraction.
        
        Args:
            article_meta: Article metadata dictionary
            
        Returns:
            Search query string
        """
        # Try using Gemini AI for intelligent query generation
        if self.genai_client:
            try:
                prompt = f"""Given the article information below, extract the main topic and generate an unbiased news search query to find articles on the same event/topic.

Article Information:
Title: {article_meta.get('title', '')}
Description: {article_meta.get('description', '')}
Tags: {', '.join(article_meta.get('tags', []) or [])}

Query guidelines:
- Avoid including biased or opinionated terms in the query
- Ensure the query is relevant to recent news articles (use relevant keywords)
- If a query is too specific, it may yield no results
- Focus on the core event/topic, not opinions about it

Return ONLY the search query as a short string of keywords."""

                response = self.genai_client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=prompt,
                    config={"response_mime_type": "text/plain"}
                )
                
                query = response.text.strip()
                if query:
                    print(f"Generated query using Gemini: {query}")
                    return query
            except Exception as e:
                print(f"Warning: Gemini query generation failed: {e}")
        
        # Fallback: simple extraction from title and tags
        parts = [
            article_meta.get("title", ""),
            " ".join(article_meta.get("tags", []) or [])
        ]
        query = " ".join([p.strip() for p in parts if p]).strip()
        print(f"Using fallback query: {query}")
        return query
    
    def find_related_articles(self, seed_url: str, max_results: int = 20) -> List[Dict]:
        """
        Find related articles based on a seed article URL.
        
        Args:
            seed_url: URL of the seed article
            max_results: Maximum number of related articles to return
            
        Returns:
            List of article dictionaries with url, title, text, source, publishedAt, etc.
        """
        # Extract seed article
        try:
            seed_meta = self.extract_article(seed_url)
        except Exception as e:
            print(f"Error extracting seed article: {e}")
            return []
        
        # Generate search query
        query = self.generate_search_query(seed_meta)
        if not query:
            print("Could not generate search query")
            return []
        
        # Search for related articles using NewsAPI
        try:
            response = self.api.get_everything(
                q=query,
                language='en',
                page_size=min(100, max_results),
                sort_by='relevancy'
            )
        except Exception as e:
            print(f"NewsAPI error: {e}")
            return []
        
        articles = []
        for article_data in response.get("articles", [])[:max_results]:
            url = article_data.get("url")
            if not url:
                continue
            
            try:
                # Extract full article text
                meta = self.extract_article(url)
                
                # Merge with NewsAPI metadata
                meta["title"] = meta.get("title") or article_data.get("title") or ""
                meta["source"] = article_data.get("source", {}).get("name")
                
                # Parse published date
                pub_date = article_data.get("publishedAt") or meta.get("publishedAt")
                try:
                    if pub_date:
                        meta["publishedAt"] = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                    else:
                        meta["publishedAt"] = None
                except Exception:
                    meta["publishedAt"] = None
                
                # Add word count
                meta["length_words"] = len(meta["text"].split()) if meta["text"] else 0
                
                articles.append(meta)
                
            except Exception as e:
                print(f"Failed to extract article from {url}: {e}")
                continue
        
        print(f"Successfully fetched {len(articles)} related articles")
        return articles
