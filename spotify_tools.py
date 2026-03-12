"""
spotify_tools.py
LangChain tools that wrap the Spotify Web API for searching podcasts and audiobooks.
"""

import os
from typing import Any
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


def _get_spotify_client() -> spotipy.Spotify:
    """Initialise a Spotify client using client-credentials flow (no user login needed)."""
    auth_manager = SpotifyClientCredentials(
        client_id=os.environ["SPOTIFY_CLIENT_ID"],
        client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def _format_show(item: dict[str, Any], rank: int) -> dict[str, Any]:
    """Extract the fields we care about from a Spotify 'show' object (podcast)."""
    return {
        "rank": rank,
        "type": "podcast",
        "id": item.get("id", ""),
        "name": item.get("name", ""),
        "publisher": item.get("publisher", ""),
        "description": item.get("description", ""),
        "total_episodes": item.get("total_episodes", 0),
        "languages": item.get("languages", []),
        "external_url": item.get("external_urls", {}).get("spotify", ""),
        "explicit": item.get("explicit", False),
    }


def _format_audiobook(item: dict[str, Any], rank: int) -> dict[str, Any]:
    """Extract the fields we care about from a Spotify 'audiobook' object."""
    authors = [a.get("name", "") for a in item.get("authors", [])]
    narrators = [n.get("name", "") for n in item.get("narrators", [])]
    return {
        "rank": rank,
        "type": "audiobook",
        "id": item.get("id", ""),
        "name": item.get("name", ""),
        "authors": authors,
        "narrators": narrators,
        "description": item.get("description", ""),
        "total_chapters": item.get("total_chapters", 0),
        "languages": item.get("languages", []),
        "external_url": item.get("external_urls", {}).get("spotify", ""),
        "explicit": item.get("explicit", False),
        "edition": item.get("edition", ""),
    }


@tool
def search_spotify_podcasts(query: str) -> list[dict[str, Any]]:
    """
    Search Spotify for the top 10 podcasts matching the given query.

    Args:
        query: Topic or keywords to search for (e.g. "machine learning", "true crime").

    Returns:
        A list of up to 10 podcast dicts, each containing name, publisher,
        description, total_episodes, languages, and a Spotify URL.
    """
    sp = _get_spotify_client()
    results = sp.search(q=query, type="show", limit=10, market="US")
    items = results.get("shows", {}).get("items", [])
    return [_format_show(item, rank=i + 1) for i, item in enumerate(items) if item]


@tool
def search_spotify_audiobooks(query: str) -> list[dict[str, Any]]:
    """
    Search Spotify for the top 3 audiobooks matching the given query.

    Args:
        query: Topic or keywords to search for (e.g. "stoicism", "finance").

    Returns:
        A list of up to 3 audiobook dicts, each containing name, authors,
        narrators, description, total_chapters, languages, and a Spotify URL.
    """
    sp = _get_spotify_client()
    results = sp.search(q=query, type="audiobook", limit=3, market="US")
    items = results.get("audiobooks", {}).get("items", [])
    return [_format_audiobook(item, rank=i + 1) for i, item in enumerate(items) if item]
