from neo4j import GraphDatabase
from langchain_community.embeddings import Embedding
from langchain_community.cache import QueryCache
from config import OPENAI_API_KEY, PickleCacheActivated

class UnifiedDataClass:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.embedding_index = Embedding()
        self.query_cache = QueryCache()

    def close(self):
        self.driver.close()

    def create_node(self, label, properties):
        with self.driver.session() as session:
            result = session.run(f"CREATE (n:{label} $properties) RETURN n", properties=properties)
            return result.single()[0]

    def find_nodes_by_property(self, label, property_name, property_value):
        with self.driver.session() as session:
            result = session.run(f"MATCH (n:{label}) WHERE n.{property_name} = $value RETURN n", value=property_value)
            return [record["n"] for record in result]

    def add_embedding(self, data):
        return self.embedding_index.add(data)

    def search_by_embedding(self, query):
        return self.embedding_index.search(query)

    def cache_query(self, query, result):
        if PickleCacheActivated:
            self.query_cache.set(query, result)

    def get_cached_query(self, query):
        if PickleCacheActivated:
            return self.query_cache.get(query)

