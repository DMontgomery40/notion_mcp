from mcp.server import Server
from mcp.types import (
    Resource, 
    Tool,
    TextContent,
    EmbeddedResource
)
from pydantic import AnyUrl
import os
import json
from datetime import datetime
import httpx
from typing import Any, Sequence, Dict, List, Optional
from dotenv import load_dotenv
from pathlib import Path
import logging
import asyncio

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('notion_mcp')

class MemoryGraph:
    """Memory management using a simple knowledge graph."""
    def __init__(self):
        self.entities: Dict[str, dict] = {}
        self.relations: List[dict] = []
    
    def add_entity(self, name: str, entity_type: str, observations: Optional[List[str]] = None) -> None:
        if observations is None:
            observations = []
        self.entities[name] = {
            "type": entity_type,
            "observations": observations
        }
    
    def add_relation(self, from_entity: str, relation_type: str, to_entity: str) -> None:
        if from_entity not in self.entities or to_entity not in self.entities:
            raise ValueError("Both entities must exist in the graph")
        
        self.relations.append({
            "from": from_entity,
            "type": relation_type,
            "to": to_entity
        })
    
    def add_observation(self, entity_name: str, observation: str) -> None:
        if entity_name not in self.entities:
            raise ValueError(f"Entity {entity_name} not found")
        
        self.entities[entity_name]["observations"].append(observation)
    
    def get_entity(self, name: str) -> Optional[dict]:
        return self.entities.get(name)
    
    def get_related_entities(self, entity_name: str) -> List[dict]:
        related = []
        for relation in self.relations:
            if relation["from"] == entity_name:
                related.append({
                    "entity": self.entities[relation["to"]],
                    "relation": relation["type"],
                    "direction": "outgoing"
                })
            elif relation["to"] == entity_name:
                related.append({
                    "entity": self.entities[relation["from"]],
                    "relation": relation["type"],
                    "direction": "incoming"
                })
        return related

    def to_json(self) -> str:
        return json.dumps({
            "entities": self.entities,
            "relations": self.relations
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MemoryGraph':
        data = json.loads(json_str)
        graph = cls()
        graph.entities = data["entities"]
        graph.relations = data["relations"]
        return graph

# Find and load .env file from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
if not env_path.exists():
    raise FileNotFoundError(f"No .env file found at {env_path}")
load_dotenv(env_path)

# Initialize server and memory
server = Server("notion-todo")
memory_graph = MemoryGraph()

# Configuration with validation
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

if not NOTION_API_KEY:
    raise ValueError("NOTION_API_KEY not found in .env file")
if not DATABASE_ID:
    raise ValueError("NOTION_DATABASE_ID not found in .env file")

NOTION_VERSION = "2022-06-28"
NOTION_BASE_URL = "https://api.notion.com/v1"

# Notion API headers
headers = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Content-Type": "application/json",
    "Notion-Version": NOTION_VERSION
}

async def fetch_todos() -> dict:
    """Fetch todos from Notion database"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{NOTION_BASE_URL}/databases/{DATABASE_ID}/query",
            headers=headers,
            json={
                "sorts": [
                    {
                        "timestamp": "created_time",
                        "direction": "descending"
                    }
                ]
            }
        )
        response.raise_for_status()
        return response.json()

async def create_todo(task: str, when: str) -> dict:
    """Create a new todo in Notion and add to memory graph"""
    async with httpx.AsyncClient() as client:
        # Create todo in Notion
        response = await client.post(
            f"{NOTION_BASE_URL}/pages",
            headers=headers,
            json={
                "parent": {"database_id": DATABASE_ID},
                "properties": {
                    "Task": {
                        "type": "title",
                        "title": [{"type": "text", "text": {"content": task}}]
                    },
                    "When": {
                        "type": "select",
                        "select": {"name": when}
                    },
                    "Checkbox": {
                        "type": "checkbox",
                        "checkbox": False
                    }
                }
            }
        )
        response.raise_for_status()
        result = response.json()
        
        # Add to memory graph
        task_id = result["id"]
        memory_graph.add_entity(task_id, "todo", [f"Task: {task}", f"When: {when}", "Status: pending"])
        return result

async def complete_todo(page_id: str) -> dict:
    """Mark a todo as complete in both Notion and memory graph"""
    async with httpx.AsyncClient() as client:
        # Update in Notion
        response = await client.patch(
            f"{NOTION_BASE_URL}/pages/{page_id}",
            headers=headers,
            json={
                "properties": {
                    "Checkbox": {
                        "type": "checkbox",
                        "checkbox": True
                    }
                }
            }
        )
        response.raise_for_status()
        result = response.json()
        
        # Update in memory graph
        if memory_graph.get_entity(page_id):
            memory_graph.add_observation(page_id, "Status: completed")
        
        return result

async def create_task_relationship(task_id: str, related_task_id: str, relationship_type: str) -> None:
    """Create a relationship between two tasks."""
    if task_id == related_task_id:
        raise ValueError("Cannot create a relationship between a task and itself")
        
    valid_relationships = ["blocks", "related_to", "parent_of", "followed_by"]
    if relationship_type not in valid_relationships:
        raise ValueError(f"Invalid relationship type. Must be one of: {valid_relationships}")
    
    # Add relationship to memory graph
    memory_graph.add_relation(task_id, relationship_type, related_task_id)
    
    # Add observations to both tasks
    memory_graph.add_observation(task_id, f"{relationship_type} {related_task_id}")
    if relationship_type == "blocks":
        memory_graph.add_observation(related_task_id, f"blocked_by {task_id}")
    elif relationship_type == "parent_of":
        memory_graph.add_observation(related_task_id, f"child_of {task_id}")
    elif relationship_type == "followed_by":
        memory_graph.add_observation(related_task_id, f"follows {task_id}")
    else:
        memory_graph.add_observation(related_task_id, f"related_to {task_id}")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="add_todo",
            description="Add a new todo item",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The todo task description"
                    },
                    "when": {
                        "type": "string",
                        "description": "When the task should be done (today or later)",
                        "enum": ["today", "later"]
                    }
                },
                "required": ["task", "when"]
            }
        ),
        Tool(
            name="show_all_todos",
            description="Show all todo items from Notion",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="show_today_todos",
            description="Show today's todo items from Notion",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="complete_todo",
            description="Mark a todo item as complete",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The ID of the todo task to mark as complete"
                    }
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="show_memory",
            description="Show the current state of the memory graph",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="link_tasks",
            description="Create a relationship between two tasks",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The ID of the first task"
                    },
                    "related_task_id": {
                        "type": "string",
                        "description": "The ID of the second task"
                    },
                    "relationship_type": {
                        "type": "string",
                        "description": "Type of relationship between tasks",
                        "enum": ["blocks", "related_to", "parent_of", "followed_by"]
                    }
                },
                "required": ["task_id", "related_task_id", "relationship_type"]
            }
        ),
        Tool(
            name="show_task_network",
            description="Show all tasks related to a given task",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The ID of the task to show relationships for"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "How many levels of relationships to show (default: 1)",
                        "default": 1
                    }
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="get_blocked_tasks",
            description="Show all tasks that are blocked by other uncompleted tasks",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_task_history",
            description="Get the complete history of a task from memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The ID of the task to get history for"
                    }
                },
                "required": ["task_id"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | EmbeddedResource]:
    """Handle tool calls for todo management"""
    if name == "add_todo":
        if not isinstance(arguments, dict):
            raise ValueError("Invalid arguments")
            
        task = arguments.get("task")
        when = arguments.get("when", "later")
        
        if not task:
            raise ValueError("Task is required")
        if when not in ["today", "later"]:
            raise ValueError("When must be 'today' or 'later'")
            
        try:
            result = await create_todo(task, when)
            return [
                TextContent(
                    type="text",
                    text=f"Added todo: {task} (scheduled for {when})\nTask ID: {result['id']}"
                )
            ]
        except httpx.HTTPError as e:
            logger.error(f"Notion API error: {str(e)}")
            return [
                TextContent(
                    type="text",
                    text=f"Error adding todo: {str(e)}\nPlease make sure your Notion integration is properly set up."
                )
            ]
            
    elif name in ["show_all_todos", "show_today_todos"]:
        try:
            todos = await fetch_todos()
            formatted_todos = []
            for todo in todos.get("results", []):
                props = todo["properties"]
                formatted_todo = {
                    "id": todo["id"],
                    "task": props["Task"]["title"][0]["text"]["content"] if props["Task"]["title"] else "",
                    "completed": props["Checkbox"]["checkbox"],
                    "when": props["When"]["select"]["name"] if props["When"]["select"] else "unknown",
                    "created": todo["created_time"]
                }
                
                if name == "show_today_todos" and formatted_todo["when"].lower() != "today":
                    continue
                    
                formatted_todos.append(formatted_todo)
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps(formatted_todos, indent=2)
                )
            ]
        except httpx.HTTPError as e:
            logger.error(f"Notion API error: {str(e)}")
            return [
                TextContent(
                    type="text",
                    text=f"Error fetching todos: {str(e)}"
                )
            ]
    
    elif name == "complete_todo":
        if not isinstance(arguments, dict):
            raise ValueError("Invalid arguments")
            
        task_id = arguments.get("task_id")
        if not task_id:
            raise ValueError("Task ID is required")
            
        try:
            result = await complete_todo(task_id)
            return [
                TextContent(
                    type="text",
                    text=f"Marked todo as complete (ID: {task_id})"
                )
            ]
        except httpx.HTTPError as e:
            logger.error(f"Notion API error: {str(e)}")
            return [
                TextContent(
                    type="text",
                    text=f"Error completing todo: {str(e)}"
                )
            ]
    
    elif name == "show_memory":
        return [
            TextContent(
                type="text",
                text=memory_graph.to_json()
            )
        ]
    
    elif name == "link_tasks":
        task_id = arguments.get("task_id")
        related_task_id = arguments.get("related_task_id")
        relationship_type = arguments.get("relationship_type")
        
        if not all([task_id, related_task_id, relationship_type]):
            raise ValueError("Missing required arguments")
            
        try:
            await create_task_relationship(task_id, related_task_id, relationship_type)
            return [
                TextContent(
                    type="text",
                    text=f"Created {relationship_type} relationship between tasks {task_id} and {related_task_id}"
                )
            ]
        except ValueError as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error creating relationship: {str(e)}"
                )
            ]
    
    elif name == "show_task_network":
        task_id = arguments.get("task_id")
        depth = arguments.get("depth", 1)
        
        if not task_id:
            raise ValueError("Task ID is required")
            
        def get_network_recursive(current_id: str, current_depth: int, visited: set) -> dict:
            if current_depth > depth or current_id in visited:
                return {}
                
            visited.add(current_id)
            entity = memory_graph.get_entity(current_id)
            if not entity:
                return {}
                
            related = memory_graph.get_related_entities(current_id)
            network = {
                "task": entity,
                "relationships": []
            }
            
            for rel in related:
                rel_id = rel["entity"]["name"] if "name" in rel["entity"] else None
                if rel_id and rel_id not in visited:
                    network["relationships"].append({
                        "type": rel["relation"],
                        "direction": rel["direction"],
                        "related_task": get_network_recursive(rel_id, current_depth + 1, visited)
                    })
            
            return network
            
        network = get_network_recursive(task_id, 0, set())
        return [
            TextContent(
                type="text",
                text=json.dumps(network, indent=2)
            )
        ]
    
    elif name == "get_blocked_tasks":
        blocked_tasks = []
        todos = await fetch_todos()
        
        # Create a mapping of task IDs to their completion status
        task_status = {
            todo["id"]: todo["properties"]["Checkbox"]["checkbox"]
            for todo in todos.get("results", [])
        }
        
        # Find all blocking relationships
        for todo in todos.get("results", []):
            task_id = todo["id"]
            related = memory_graph.get_related_entities(task_id)
            
            for rel in related:
                if rel["relation"] == "blocks" and rel["direction"] == "outgoing":
                    blocker_id = task_id
                    blocked_id = rel["entity"]["name"] if "name" in rel["entity"] else None
                    
                    if blocked_id and not task_status.get(blocker_id, True):
                        blocked_tasks.append({
                            "blocked_task": blocked_id,
                            "blocked_by": blocker_id,
                            "blocker_status": "incomplete"
                        })
        
        return [
            TextContent(
                type="text",
                text=json.dumps(blocked_tasks, indent=2)
            )
        ]
    
    elif name == "get_task_history":
        task_id = arguments.get("task_id")
        if not task_id:
            raise ValueError("Task ID is required")
        
        entity = memory_graph.get_entity(task_id)
        if not entity:
            return [
                TextContent(
                    type="text",
                    text=f"No history found for task {task_id}"
                )
            ]
        
        return [
            TextContent(
                type="text",
                text=json.dumps({
                    "task_id": task_id,
                    "history": entity["observations"],
                    "related": memory_graph.get_related_entities(task_id)
                }, indent=2)
            )
        ]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    """Main entry point for the server"""
    from mcp.server.stdio import stdio_server
    
    if not NOTION_API_KEY or not DATABASE_ID:
        raise ValueError("NOTION_API_KEY and NOTION_DATABASE_ID environment variables are required")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
