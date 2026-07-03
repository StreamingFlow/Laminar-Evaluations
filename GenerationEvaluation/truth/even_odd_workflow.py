from dispel4py.base import ProducerPE, GenericPE, IterativePE
from dispel4py.workflow_graph import WorkflowGraph


class NumberProducer(ProducerPE):
    def __init__(self):
        ProducerPE.__init__(self)

    def _process(self, inputs):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class Divideby2(IterativePE):

    def __init__(self, compare):
        IterativePE.__init__(self)
        self.compare = compare

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class PairProducer(GenericPE):

    def __init__(self):
        GenericPE.__init__(self)
        self._add_input("odd")
        self._add_input("even")
        self._add_output("output")
        self.list_odd = []
        self.list_even = []

    def _process(self, inputs):
        raise NotImplementedError('   business logic is intentionally unimplemented')

    def _postprocess(self):
        raise NotImplementedError('   business logic is intentionally unimplemented')


producer = NumberProducer()
filter_even = Divideby2(0)
filter_odd = Divideby2(1)
pair = PairProducer()

graph = WorkflowGraph()
graph.connect(producer, 'output', filter_even, 'input')
graph.connect(producer, 'output', filter_odd, 'input')
graph.connect(filter_even, 'output', pair, 'even')
graph.connect(filter_odd, 'output', pair, 'odd')
