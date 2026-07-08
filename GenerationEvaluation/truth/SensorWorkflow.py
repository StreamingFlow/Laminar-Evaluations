from dispel4py.base import IterativePE, ConsumerPE, ProducerPE
from dispel4py.workflow_graph import WorkflowGraph


class ReadSensorDataPE(ProducerPE):
    def __init__(self):
        ProducerPE.__init__(self)

    def _process(self, inputs):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class NormalizeDataPE(IterativePE):
    def __init__(self):
        IterativePE.__init__(self)

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class AnomalyDetectionPE(IterativePE):
    def __init__(self):
        IterativePE.__init__(self)

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class AlertingPE(IterativePE):
    def __init__(self):
        IterativePE.__init__(self)

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class AggregateDataPE(ConsumerPE):
    def __init__(self):
        ConsumerPE.__init__(self)
        self.temperatures = []

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


graph = WorkflowGraph()

read = ReadSensorDataPE()
normalize_data = NormalizeDataPE()
alerting = AlertingPE()
anomaly_detection = AnomalyDetectionPE()
aggregate_data = AggregateDataPE()

graph.connect(read, 'output', normalize_data, 'input')
graph.connect(normalize_data, 'output', anomaly_detection, 'input')
graph.connect(anomaly_detection, 'output', alerting, 'input')
graph.connect(alerting, 'output', aggregate_data, 'input')