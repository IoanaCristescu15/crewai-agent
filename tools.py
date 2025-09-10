from __future__ import annotations

import os
import json
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import requests
import trafilatura
from pypdf import PdfReader
from crewai.tools import BaseTool


class UrlReaderTool(BaseTool):
    """Fetch a single URL and extract readable text using trafilatura.

    Inputs:
      - url: exact URL to fetch
    Output:
      - extracted readable text or empty string if not retrievable
    """

    name: str = "url_reader"
    description: str = (
        "Fetch a single exact URL and return readable text. No generic search."
    )

    def _run(self, url: str) -> str:
        if not url or not isinstance(url, str):
            return ""
        
        # Try trafilatura first as it handles user-agent and other headers automatically
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded, include_comments=False) or ""
                if text.strip():
                    return text.strip()
        except Exception:
            pass

        # Fallback to requests with proper headers
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            resp = requests.get(url, timeout=20, headers=headers)
            resp.raise_for_status()
            return resp.text.strip() if isinstance(resp.text, str) else ""
        except Exception:
            return ""


class PdfReaderTool(BaseTool):
    """Extract text from a local PDF using pypdf."""

    name: str = "pdf_reader"
    description: str = "Extract text content from a local PDF file path."

    def _run(self, file_path: str) -> str:
        if not file_path or not os.path.exists(file_path):
            return ""
        try:
            reader = PdfReader(file_path)
            pages_text = []
            for page in reader.pages:
                try:
                    pages_text.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n".join(t.strip() for t in pages_text if t).strip()
        except Exception:
            return ""


class PasteTool(BaseTool):
    """Accept raw pasted text from the CLI and return it unchanged."""

    name: str = "paste_tool"
    description: str = "Accept raw text provided on the CLI and return it."

    def _run(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        return text.strip()


class WebSearchTool(BaseTool):
    """Search the web for information using DuckDuckGo (no API key required)."""

    name: str = "web_search"
    description: str = "Search the web for current information on any topic. Returns search results with titles, snippets, and URLs."

    def _run(self, query: str) -> str:
        if not query or not isinstance(query, str):
            return ""
        
        try:
            # Use DuckDuckGo instant answer API (no API key required)
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if data.get('Abstract'):
                results.append(f"Summary: {data['Abstract']}")
            if data.get('AbstractURL'):
                results.append(f"Source: {data['AbstractURL']}")
            
            # Add related topics
            if data.get('RelatedTopics'):
                results.append("\nRelated Topics:")
                for topic in data['RelatedTopics'][:3]:  # Limit to 3 topics
                    if isinstance(topic, dict) and topic.get('Text'):
                        results.append(f"- {topic['Text']}")
            
            return "\n".join(results) if results else "No search results found."
            
        except Exception as e:
            return f"Search failed: {str(e)}"


class EmailDraftTool(BaseTool):
    """Draft professional emails with proper formatting and tone."""

    name: str = "email_draft"
    description: str = "Draft professional emails with appropriate tone and formatting. Input: recipient, subject, purpose, and key points."

    def _run(self, input_data: str) -> str:
        if not input_data or not isinstance(input_data, str):
            return ""
        
        try:
            # Parse input (expecting JSON or structured text)
            if input_data.strip().startswith('{'):
                data = json.loads(input_data)
            else:
                # Simple parsing for non-JSON input
                lines = input_data.strip().split('\n')
                data = {
                    'recipient': lines[0] if len(lines) > 0 else 'Recipient',
                    'subject': lines[1] if len(lines) > 1 else 'Subject',
                    'purpose': lines[2] if len(lines) > 2 else 'Purpose',
                    'key_points': lines[3:] if len(lines) > 3 else []
                }
            
            recipient = data.get('recipient', 'Recipient')
            subject = data.get('subject', 'Subject')
            purpose = data.get('purpose', 'Purpose')
            key_points = data.get('key_points', [])
            
            # Generate email draft
            email = f"""To: {recipient}
Subject: {subject}

Hi {recipient.split()[0] if recipient else 'there'},

{purpose}

"""
            
            if key_points:
                email += "Key points:\n"
                for point in key_points:
                    if point.strip():
                        email += f"• {point.strip()}\n"
                email += "\n"
            
            email += """Best regards,
Ioană

---
[This is a draft - please review before sending]"""
            
            return email
            
        except Exception as e:
            return f"Email draft failed: {str(e)}"


class CodeAnalysisTool(BaseTool):
    """Analyze code for issues, suggest improvements, and explain functionality."""

    name: str = "code_analysis"
    description: str = "Analyze code for bugs, performance issues, and suggest improvements. Also explain what code does."

    def _run(self, code: str) -> str:
        if not code or not isinstance(code, str):
            return ""
        
        try:
            analysis = []
            
            # Basic code analysis
            lines = code.split('\n')
            analysis.append(f"Code Analysis Report:")
            analysis.append(f"Lines of code: {len(lines)}")
            analysis.append(f"Characters: {len(code)}")
            
            # Look for common issues
            issues = []
            
            # Check for TODO comments
            todos = [line.strip() for line in lines if 'TODO' in line.upper() or 'FIXME' in line.upper()]
            if todos:
                issues.append(f"TODOs/FIXMEs found: {len(todos)}")
                for todo in todos[:3]:  # Show first 3
                    issues.append(f"  - {todo}")
            
            # Check for long lines
            long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 100]
            if long_lines:
                issues.append(f"Long lines (>100 chars): {len(long_lines)} lines")
            
            # Check for potential issues
            if 'print(' in code and 'logging' not in code.lower():
                issues.append("Consider using logging instead of print statements")
            
            if 'except:' in code:
                issues.append("Consider specifying exception types instead of bare except")
            
            if 'eval(' in code or 'exec(' in code:
                issues.append("WARNING: eval/exec usage detected - security risk")
            
            # Language detection
            if 'def ' in code and 'class ' in code:
                language = "Python"
            elif 'function ' in code and '{' in code:
                language = "JavaScript"
            elif '#include' in code:
                language = "C/C++"
            else:
                language = "Unknown"
            
            analysis.append(f"Detected language: {language}")
            
            if issues:
                analysis.append("\nPotential Issues:")
                for issue in issues:
                    analysis.append(f"• {issue}")
            else:
                analysis.append("\nNo obvious issues detected.")
            
            # Suggest improvements
            suggestions = []
            if len(lines) > 50:
                suggestions.append("Consider breaking into smaller functions")
            if 'password' in code.lower() or 'secret' in code.lower():
                suggestions.append("Ensure sensitive data is properly secured")
            if 'http://' in code:
                suggestions.append("Consider using HTTPS for security")
            
            if suggestions:
                analysis.append("\nSuggestions:")
                for suggestion in suggestions:
                    analysis.append(f"• {suggestion}")
            
            return "\n".join(analysis)
            
        except Exception as e:
            return f"Code analysis failed: {str(e)}"


class CalendarTool(BaseTool):
    """Manage calendar events and scheduling."""

    name: str = "calendar_manager"
    description: str = "Create, list, and manage calendar events. Input: action (create/list), event details."

    def _run(self, input_data: str) -> str:
        if not input_data or not isinstance(input_data, str):
            return ""
        
        try:
            # Simple calendar simulation (in real implementation, would integrate with Google Calendar, Outlook, etc.)
            if input_data.strip().startswith('{'):
                data = json.loads(input_data)
            else:
                # Simple parsing
                parts = input_data.strip().split('|')
                data = {
                    'action': parts[0].strip() if len(parts) > 0 else 'list',
                    'title': parts[1].strip() if len(parts) > 1 else '',
                    'date': parts[2].strip() if len(parts) > 2 else '',
                    'time': parts[3].strip() if len(parts) > 3 else '',
                    'duration': parts[4].strip() if len(parts) > 4 else '1 hour'
                }
            
            action = data.get('action', 'list').lower()
            
            if action == 'create':
                title = data.get('title', 'New Event')
                date = data.get('date', 'today')
                time = data.get('time', '9:00 AM')
                duration = data.get('duration', '1 hour')
                
                return f"""Event Created:
Title: {title}
Date: {date}
Time: {time}
Duration: {duration}

[Note: This is a simulation. In a real implementation, this would create an actual calendar event.]"""
            
            elif action == 'list':
                # Simulate listing upcoming events
                today = datetime.now()
                events = [
                    f"• Data Science Class - {today.strftime('%Y-%m-%d')} 10:00 AM - 11:30 AM",
                    f"• ML Infrastructure Review - {(today + timedelta(days=1)).strftime('%Y-%m-%d')} 2:00 PM - 3:00 PM",
                    f"• Research Meeting - {(today + timedelta(days=3)).strftime('%Y-%m-%d')} 1:00 PM - 2:00 PM"
                ]
                
                return f"Upcoming Events:\n" + "\n".join(events)
            
            else:
                return "Available actions: 'create' or 'list'"
                
        except Exception as e:
            return f"Calendar operation failed: {str(e)}"


class ResearchTool(BaseTool):
    """Conduct research on academic and technical topics."""

    name: str = "research_assistant"
    description: str = "Research academic papers, technical documentation, and current trends in data science and ML."

    def _run(self, topic: str) -> str:
        if not topic or not isinstance(topic, str):
            return ""
        
        try:
            # Simulate research by providing structured information
            # In a real implementation, this would search academic databases, arXiv, etc.
            
            research_areas = {
                'machine learning': {
                    'trends': ['Large Language Models', 'Federated Learning', 'AutoML', 'MLOps'],
                    'papers': ['Attention Is All You Need', 'BERT: Pre-training of Deep Bidirectional Transformers'],
                    'tools': ['TensorFlow', 'PyTorch', 'Scikit-learn', 'Hugging Face']
                },
                'data science': {
                    'trends': ['Data Engineering', 'Real-time Analytics', 'Data Mesh', 'MLOps'],
                    'papers': ['The Data Science Process', 'CRISP-DM Methodology'],
                    'tools': ['Pandas', 'NumPy', 'Apache Spark', 'Docker']
                },
                'ml infrastructure': {
                    'trends': ['MLOps', 'Model Serving', 'Feature Stores', 'ML Observability'],
                    'papers': ['Hidden Technical Debt in Machine Learning Systems', 'MLOps: Continuous delivery and automation pipelines'],
                    'tools': ['Kubeflow', 'MLflow', 'Seldon', 'Weights & Biases']
                }
            }
            
            topic_lower = topic.lower()
            found_area = None
            
            for area, info in research_areas.items():
                if any(keyword in topic_lower for keyword in area.split()):
                    found_area = area
                    break
            
            if not found_area:
                found_area = 'data science'  # Default
            
            info = research_areas[found_area]
            
            result = f"Research Report: {topic}\n"
            result += f"Focus Area: {found_area.title()}\n\n"
            
            result += "Current Trends:\n"
            for trend in info['trends']:
                result += f"• {trend}\n"
            
            result += "\nKey Papers:\n"
            for paper in info['papers']:
                result += f"• {paper}\n"
            
            result += "\nRecommended Tools:\n"
            for tool in info['tools']:
                result += f"• {tool}\n"
            
            result += f"\n[Note: This is a simulated research report. For real research, integrate with arXiv, Google Scholar, or other academic databases.]"
            
            return result
            
        except Exception as e:
            return f"Research failed: {str(e)}"


