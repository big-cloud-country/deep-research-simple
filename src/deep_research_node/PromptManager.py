#!/usr/bin/env python3
"""
Prompt Manager for versioned prompt templates.

This module provides a system for managing versioned prompts with metadata,
enabling A/B testing, performance tracking, and seamless integration with
OpenTelemetry tracing.
"""

import os
import yaml
import hashlib
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import frontmatter


@dataclass
class PromptTemplate:
    """Represents a versioned prompt template with metadata."""
    
    name: str
    version: str
    template: str
    author: str
    date: str
    changes: str
    tags: List[str]
    model_hints: Dict[str, Any]
    content_hash: str
    file_path: str
    
    def render(self, **kwargs) -> str:
        """Render the prompt template with provided variables."""
        return self.template.format(**kwargs)
    
    def get_otel_attributes(self) -> Dict[str, Any]:
        """Get OpenTelemetry attributes for this prompt."""
        attributes = {
            "prompt.name": self.name,
            "prompt.version": self.version,
            "prompt.author": self.author,
            "prompt.date": self.date,
            "prompt.hash": self.content_hash,
            "prompt.tags": ",".join(self.tags),
            "prompt.file": os.path.basename(self.file_path)
        }
        
        # Add model hints as attributes
        for key, value in self.model_hints.items():
            attributes[f"model.hint.{key}"] = value
        
        return attributes
    
    def get_langfuse_metadata(self) -> Dict[str, Any]:
        """Get Langfuse-specific metadata for enhanced UI visibility."""
        return {
            "langfuse.metadata.prompt_name": self.name,
            "langfuse.metadata.prompt_version": self.version,
            "langfuse.metadata.prompt_author": self.author,
            "langfuse.metadata.prompt_date": self.date,
            "langfuse.metadata.prompt_tags": ",".join(self.tags),
            "langfuse.metadata.prompt_hash": self.content_hash,
            # Add prompt name as a tag for easy filtering
            "langfuse.tags": f"{self.name},{self.version}," + ",".join(self.tags)
        }


class PromptManager:
    """Manages versioned prompt templates."""
    
    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize the prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt files
        """
        self.prompts_dir = prompts_dir
        self.cache: Dict[str, Dict[str, PromptTemplate]] = {}
        self.manifest: Dict[str, Any] = {}
        self._load_manifest()
        self._load_all_prompts()
    
    def _load_manifest(self) -> None:
        """Load the prompt manifest file."""
        manifest_path = os.path.join(self.prompts_dir, "manifest.yaml")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.manifest = yaml.safe_load(f)
        else:
            print(f"⚠️  No manifest found at {manifest_path}")
            self.manifest = {"prompts": {}, "settings": {}}
    
    def _load_all_prompts(self) -> None:
        """Load all prompts from the prompts directory."""
        prompts_config = self.manifest.get("prompts", {})
        
        for prompt_name, config in prompts_config.items():
            prompt_dir = os.path.join(self.prompts_dir, prompt_name)
            if not os.path.exists(prompt_dir):
                continue
            
            self.cache[prompt_name] = {}
            
            # Load all version files
            for filename in os.listdir(prompt_dir):
                if filename.endswith('.md'):
                    version = filename.replace('.md', '')
                    file_path = os.path.join(prompt_dir, filename)
                    
                    try:
                        prompt_template = self._load_prompt_file(
                            prompt_name, version, file_path
                        )
                        self.cache[prompt_name][version] = prompt_template
                    except Exception as e:
                        print(f"❌ Error loading {file_path}: {e}")
    
    def _load_prompt_file(self, name: str, version: str, file_path: str) -> PromptTemplate:
        """Load a single prompt file."""
        with open(file_path, 'r') as f:
            post = frontmatter.load(f)
        
        # Extract metadata
        metadata = post.metadata
        content = post.content.strip()
        
        # Calculate content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        
        return PromptTemplate(
            name=name,
            version=version,
            template=content,
            author=metadata.get("author", "unknown"),
            date=metadata.get("date", "unknown"),
            changes=metadata.get("changes", ""),
            tags=metadata.get("tags", []),
            model_hints=metadata.get("model_hints", {}),
            content_hash=content_hash,
            file_path=file_path
        )
    
    def get_prompt(self, name: str, version: Optional[str] = None) -> PromptTemplate:
        """
        Get a specific prompt version.
        
        Args:
            name: Name of the prompt
            version: Version to retrieve (default: latest)
            
        Returns:
            PromptTemplate instance
            
        Raises:
            ValueError: If prompt or version not found
        """
        if name not in self.cache:
            raise ValueError(f"Prompt '{name}' not found")
        
        if version is None or version == "latest":
            # Get latest version from manifest
            prompt_config = self.manifest.get("prompts", {}).get(name, {})
            version = prompt_config.get("latest", None)
            
            if not version:
                # Fallback to highest version number
                versions = list(self.cache[name].keys())
                if versions:
                    version = sorted(versions)[-1]
                else:
                    raise ValueError(f"No versions found for prompt '{name}'")
        
        if version not in self.cache[name]:
            raise ValueError(f"Version '{version}' not found for prompt '{name}'")
        
        return self.cache[name][version]
    
    def list_prompts(self) -> List[str]:
        """List all available prompt names."""
        return list(self.cache.keys())
    
    def list_versions(self, name: str) -> List[str]:
        """List all versions for a specific prompt."""
        if name not in self.cache:
            raise ValueError(f"Prompt '{name}' not found")
        
        return sorted(list(self.cache[name].keys()))
    
    def get_changelog(self, name: str) -> Dict[str, str]:
        """Get version history for a prompt."""
        if name not in self.cache:
            raise ValueError(f"Prompt '{name}' not found")
        
        changelog = {}
        for version, template in self.cache[name].items():
            changelog[version] = {
                "date": template.date,
                "author": template.author,
                "changes": template.changes
            }
        
        return changelog
    
    def compare_versions(self, name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions of a prompt."""
        prompt1 = self.get_prompt(name, version1)
        prompt2 = self.get_prompt(name, version2)
        
        return {
            "version1": version1,
            "version2": version2,
            "content_changed": prompt1.template != prompt2.template,
            "hash1": prompt1.content_hash,
            "hash2": prompt2.content_hash,
            "tags_added": list(set(prompt2.tags) - set(prompt1.tags)),
            "tags_removed": list(set(prompt1.tags) - set(prompt2.tags)),
            "model_hints_changed": prompt1.model_hints != prompt2.model_hints
        }
    
    def get_ab_test_variants(self, name: str, variants: List[str]) -> List[PromptTemplate]:
        """Get multiple prompt versions for A/B testing."""
        return [self.get_prompt(name, version) for version in variants]
    
    def reload(self) -> None:
        """Reload all prompts from disk."""
        self.cache.clear()
        self._load_manifest()
        self._load_all_prompts()


# Convenience function for quick prompt loading
def load_prompt(name: str, version: Optional[str] = None, prompts_dir: str = "prompts") -> PromptTemplate:
    """
    Quick function to load a single prompt.
    
    Args:
        name: Prompt name
        version: Version (default: latest)
        prompts_dir: Directory containing prompts
        
    Returns:
        PromptTemplate instance
    """
    manager = PromptManager(prompts_dir)
    return manager.get_prompt(name, version)