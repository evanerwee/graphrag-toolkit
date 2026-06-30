# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for codeproperty-graph core components."""

from graphrag_toolkit.codeproperty_graph.graph_diff import GraphDiff, DiffResult
from graphrag_toolkit.codeproperty_graph.manifest_manager import ManifestManager
from graphrag_toolkit.codeproperty_graph.models import CPGNode, CPGEdge, Manifest


def test_graph_diff_no_changes():
    prev = {"pkg.Foo.bar": "hash1", "pkg.Foo.baz": "hash2"}
    curr = {"pkg.Foo.bar": "hash1", "pkg.Foo.baz": "hash2"}
    result = GraphDiff.compare(prev, curr)
    assert not result.has_changes
    assert result.unchanged == 2
    assert result.summary == "+0 -0 ~0 =2"


def test_graph_diff_added():
    prev = {"pkg.Foo.bar": "h1"}
    curr = {"pkg.Foo.bar": "h1", "pkg.Foo.new_method": "h2"}
    result = GraphDiff.compare(prev, curr)
    assert result.has_changes
    assert "pkg.Foo.new_method" in result.added


def test_graph_diff_removed():
    prev = {"pkg.Foo.bar": "h1", "pkg.Foo.old": "h2"}
    curr = {"pkg.Foo.bar": "h1"}
    result = GraphDiff.compare(prev, curr)
    assert "pkg.Foo.old" in result.removed


def test_graph_diff_modified():
    prev = {"pkg.Foo.bar": "hash_v1"}
    curr = {"pkg.Foo.bar": "hash_v2"}
    result = GraphDiff.compare(prev, curr)
    assert "pkg.Foo.bar" in result.modified
    assert result.modified["pkg.Foo.bar"] == "hash_v2"


def test_graph_diff_mixed():
    prev = {"a": "1", "b": "2", "c": "3"}
    curr = {"a": "1", "b": "CHANGED", "d": "4"}
    result = GraphDiff.compare(prev, curr)
    assert result.added == {"d": "4"}
    assert result.removed == {"c": "3"}
    assert result.modified == {"b": "CHANGED"}
    assert result.unchanged == 1
    assert result.summary == "+1 -1 ~1 =1"


def test_cpg_node_stable_id():
    node = CPGNode(id="123", node_type="METHOD", full_name="pkg.MyClass.run")
    assert node.stable_id == "pkg.MyClass.run"


def test_cpg_node_stable_id_fallback():
    node = CPGNode(id="456", node_type="LITERAL")
    assert node.stable_id == "456"


def test_cpg_edge_key():
    edge = CPGEdge(source_id="1", target_id="2", edge_type="AST")
    assert edge.key == "1->AST->2"


def test_manifest_signature():
    mgr = ManifestManager.__new__(ManifestManager)
    mgr._bucket = "test"
    mgr._prefix = "cpg-exports"
    mgr._s3 = None
    sig = mgr.compute_signature({"a": "1", "b": "2"})
    assert sig.startswith("sha256:")
    # Deterministic
    assert sig == mgr.compute_signature({"b": "2", "a": "1"})


# === Schema tests ===

def test_node_type_enum():
    from graphrag_toolkit.codeproperty_graph.schema import NodeType
    assert NodeType.METHOD == "METHOD"
    assert NodeType.CALL == "CALL"
    assert NodeType.TYPE_DECL == "TYPE_DECL"
    assert len(NodeType) == 20


def test_edge_type_enum():
    from graphrag_toolkit.codeproperty_graph.schema import EdgeType
    assert EdgeType.AST == "AST"
    assert EdgeType.CFG == "CFG"
    assert EdgeType.REACHING_DEF == "REACHING_DEF"
    assert len(EdgeType) >= 14


def test_delta_relevant_types():
    from graphrag_toolkit.codeproperty_graph.schema import DELTA_RELEVANT_TYPES, NodeType
    assert NodeType.METHOD in DELTA_RELEVANT_TYPES
    assert NodeType.COMMENT not in DELTA_RELEVANT_TYPES


def test_joern_export_command():
    from graphrag_toolkit.codeproperty_graph.schema import joern_export_command
    cmd = joern_export_command("/src/myapp", output_dir="out", language="java")
    assert "joern-export" in cmd
    assert "--format json" in cmd
    assert "--language java" in cmd
    assert "/src/myapp" in cmd


# === from_joern factory tests ===

def test_cpg_node_from_joern():
    raw = {
        "id": "1001",
        "label": "METHOD",
        "properties": {
            "FULL_NAME": "com.example.Main.main",
            "SIGNATURE": "void(String[])",
            "HASH": "abc123",
            "FILENAME": "src/Main.java",
            "LINE_NUMBER": 5,
            "CODE": "public static void main(String[] args) {}",
            "IS_EXTERNAL": False,
        }
    }
    node = CPGNode.from_joern(raw)
    assert node.id == "1001"
    assert node.node_type == "METHOD"
    assert node.full_name == "com.example.Main.main"
    assert node.signature == "void(String[])"
    assert node.hash == "abc123"
    assert node.filename == "src/Main.java"
    assert node.line_number == 5
    assert node.is_external is False


def test_cpg_node_from_joern_identifier():
    raw = {
        "id": "2001",
        "label": "IDENTIFIER",
        "properties": {"NAME": "myVar", "CODE": "myVar", "LINE_NUMBER": 10}
    }
    node = CPGNode.from_joern(raw)
    assert node.node_type == "IDENTIFIER"
    assert node.name == "myVar"
    assert node.line_number == 10


def test_cpg_edge_from_joern():
    raw = {"src": "1001", "dst": "2001", "label": "AST", "properties": {"ORDER": 1}}
    edge = CPGEdge.from_joern(raw)
    assert edge.source_id == "1001"
    assert edge.target_id == "2001"
    assert edge.edge_type == "AST"
    assert edge.properties == {"ORDER": 1}


def test_cpg_edge_from_joern_cfg():
    raw = {"src": "100", "dst": "200", "label": "CFG", "properties": {}}
    edge = CPGEdge.from_joern(raw)
    assert edge.edge_type == "CFG"
    assert edge.key == "100->CFG->200"
