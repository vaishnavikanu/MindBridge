import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

sys.path.insert(0, PROJECT_ROOT)


from src.pipeline.rag_pipeline import RAGPipeline


def main():

    print("=" * 70)
    print("Initializing RAG Pipeline...")
    print("=" * 70)

    pipeline = RAGPipeline()
    pipeline.initialize()

    print("\n")

    ##########################################################
    # Graph Statistics
    ##########################################################

    print("=" * 70)
    print("GLOBAL GRAPH")
    print("=" * 70)

    global_graph = pipeline.graph_manager.get_global_graph()

    print(global_graph.get_stats())

    print("\n")

    print("=" * 70)
    print("PATIENT GRAPH")
    print("=" * 70)

    patient_graph = pipeline.graph_manager.get_patient_graph(
        "patient_001"
    )

    print(patient_graph.get_stats())

    print("\n")

    print("=" * 70)
    print("CLINICIAN GRAPH")
    print("=" * 70)

    clinician_graph = pipeline.graph_manager.get_clinician_graph(
        "clinician_001"
    )

    print(clinician_graph.get_stats())

    print("\n")

    ##########################################################
    # Graph Expansion Test
    ##########################################################

    print("=" * 70)
    print("GRAPH QUERY EXPANSION")
    print("=" * 70)

    query = "How is depression treated?"

    graphs = pipeline.graph_manager.get_graphs(
        role="patient",
        user_id="patient_001",
    )

    expanded = pipeline.graph_retriever.expand_query(
        query,
        graphs,
    )

    print(f"\nOriginal Query:\n{query}\n")

    print("Expanded Entities:")

    for entity in expanded:
        print(f" - {entity}")

    print("\n")

    ##########################################################
    # Neighbor Test
    ##########################################################

    print("=" * 70)
    print("NEIGHBORS OF 'depression'")
    print("=" * 70)

    neighbors = global_graph.get_neighbors("depression")

    if neighbors:

        for n in neighbors:
            print(f" - {n}")

    else:

        print("No neighbors found.")

    print("\nDone.")


if __name__ == "__main__":
    main()