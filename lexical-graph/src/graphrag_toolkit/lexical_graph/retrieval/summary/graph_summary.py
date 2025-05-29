# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from graphrag_toolkit.lexical_graph import GraphRAGConfig
from graphrag_toolkit.lexical_graph import to_tenant_id, TenantIdType
from graphrag_toolkit.lexical_graph.storage.graph import GraphStore, MultiTenantGraphStore
from graphrag_toolkit.lexical_graph.utils import LLMCache, LLMCacheType

from llama_index.core.prompts import PromptTemplate


GRAPH_SUMMARY_PROMPT = '''
You are an expert AI assistant specialising in knowledge graphs. Based on the provided entities and property graph path descriptions, your task is to summarize the domain, scope and uses of a knowledge graph so that it can be utilized by a Model Context Protocol client. Include a list of up to 10 different questions the graph can be used to answer. Provide an authoritative response: do not use words such as 'appears to' or 'seems'. In your response describe the graph as a 'knowledge base', not a graph.

<entities>
{entities}
</entities>

<propertyGraphPaths>
{paths}
</propertyGraphPaths>

## Response format (separate each section with a new line)

Domain: <Headline description of the domain>
Scope: <The graph covers...>
Uses: <List valuable uses>
Example questions: <10 example questions>
'''

def get_domain(s):
    for line in s.split('\n'):
        if line.startswith('Domain:'):
            return line[7:].strip()
    return None

class GraphSummary():

    def __init__(self, graph_store:GraphStore, llm:LLMCacheType=None, prompt_template=None):
        self.graph_store = graph_store
        self.llm = llm if llm and isinstance(llm, LLMCache) else LLMCache(
            llm=llm or GraphRAGConfig.extraction_llm,
            enable_cache=GraphRAGConfig.enable_cache
        )
        self.prompt_template=prompt_template or GRAPH_SUMMARY_PROMPT
        
    def _get_entities(self, tenant_id:TenantIdType):
        
        tenant_id = to_tenant_id(tenant_id)
        graph_store = MultiTenantGraphStore.wrap(
            self.graph_store,
            tenant_id
        )
        
        cypher = '''MATCH (n:`__Entity__`)-[r:`__RELATION__`]->()
        WITH n, sum(r.count) AS score ORDER BY score DESC LIMIT 100
        RETURN n.value + ' [' + n.class + ']' as entity'''
        
        results = graph_store.execute_query(cypher)
        
        return results
        
    def _get_paths(self, tenant_id:TenantIdType):
        
        tenant_id = to_tenant_id(tenant_id)
        graph_store = MultiTenantGraphStore.wrap(
            self.graph_store,
            tenant_id
        )
        
        cypher = '''MATCH (n:`__SYS_Class__`) 
        WITH n, n.count AS score ORDER BY score DESC LIMIT 10
        MATCH p=(n)-[r]->()
        WITH nodes(p) AS nodes, relationships(p) AS rels, r.count AS score ORDER BY score DESC LIMIT 100
        RETURN '(' + nodes[0].value + ')-[' + rels[0].value + ']->(' + nodes[1].value + ')' AS path'''
        
        results = graph_store.execute_query(cypher)
        
        return results
        
    def _generate_summary(self, entities:str, paths:str):
        return self.llm.predict(
            PromptTemplate(template=self.prompt_template),
            entities=entities,
            paths=paths
        )
        
    def create_graph_summary(self, tenant_id:TenantIdType=None):
        entities = '\n'.join([
            result['entity'] for result in self._get_entities(tenant_id)
        ])
        
        paths = '\n'.join([
            result['path'] for result in self._get_paths(tenant_id)
        ])

        if entities and paths:
            return self._generate_summary(entities, paths)
        else:
            return None