"""Memory management for notion_mcp using a simple knowledge graph."""

from typing import Dict, List, Optional
import json

class MemoryGraph:
    def __init__(self):
        self.entities: Dict[str, dict] = {}
        self.relations: List[dict] = []
    
    def add_entity(self, name: str, entity_type: str, observations: Optional[List[str]] = None) -> None:
        """Add an entity to the graph."""
        if observations is None:
            observations = []
        self.entities[name] = {
            "type": entity_type,
            "observations": observations
        }
    
    def add_relation(self, from_entity: str, relation_type: str, to_entity: str) -> None:
        """Add a relation between entities."""
        if from_entity not in self.entities or to_entity not in self.entities:
            raise ValueError("Both entities must exist in the graph")
        
        self.relations.append({
            "from": from_entity,
            "type": relation_type,
            "to": to_entity
        })
    
    def add_observation(self, entity_name: str, observation: str) -> None:
        """Add an observation to an existing entity."""
        if entity_name not in self.entities:
            raise ValueError(f"Entity {entity_name} not found")
        
        self.entities[entity_name]["observations"].append(observation)
    
    def get_entity(self, name: str) -> Optional[dict]:
        """Get an entity by name."""
        return self.entities.get(name)
    
    def get_related_entities(self, entity_name: str) -> List[dict]:
        """Get all entities related to the given entity."""
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
        """Serialize the graph to JSON."""
        return json.dumps({
            "entities": self.entities,
            "relations": self.relations
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MemoryGraph':
        """Create a MemoryGraph instance from JSON."""
        data = json.loads(json_str)
        graph = cls()
        graph.entities = data["entities"]
        graph.relations = data["relations"]
        return graph
