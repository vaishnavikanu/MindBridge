from src.graph.graph_store import GraphStore


class GraphManager:

    def __init__(self):

        self.global_graph = GraphStore("global")

        self.patient_graphs = {}

        self.clinician_graphs = {}

    #####################################################

    def get_global_graph(self):

        return self.global_graph

    #####################################################

    def get_patient_graph(
        self,
        patient_id: str,
    ):

        if patient_id not in self.patient_graphs:

            self.patient_graphs[patient_id] = GraphStore(
                f"patient_{patient_id}"
            )

        return self.patient_graphs[patient_id]

    #####################################################

    def get_clinician_graph(
        self,
        clinician_id: str,
    ):

        if clinician_id not in self.clinician_graphs:

            self.clinician_graphs[clinician_id] = GraphStore(
                f"clinician_{clinician_id}"
            )

        return self.clinician_graphs[clinician_id]

    #####################################################

    def get_graphs(
        self,
        role: str,
        user_id: str,
    ):

        if role == "patient":

            return [
                self.global_graph,
                self.get_patient_graph(user_id),
            ]

        elif role == "clinician":

            return [
                self.global_graph,
                self.get_clinician_graph(user_id),
            ]

        else:

            raise ValueError(role)