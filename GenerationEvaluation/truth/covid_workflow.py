from dispel4py.base import create_iterative_chain, IterativePE, ProducerPE, GenericPE
from dispel4py.workflow_graph import WorkflowGraph

class DataProducer(GenericPE):
    def __init__(self, url):
        GenericPE.__init__(self)
        self.url = url
        self._add_output('output')

    def _process(self, inputs):
        raise NotImplementedError('   business logic is intentionally unimplemented')

class DataProcessor(IterativePE):
    def __init__(self):
        IterativePE.__init__(self)

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')

class DataVisualizer(GenericPE):
    def __init__(self):
        GenericPE.__init__(self)
        self._add_input('input')
        self._add_output('output')
        self.inputconnections['input']["grouping"] = "global"
        self.results = {}
        self.results['dates']=[]
        self.results['new_cases']=[]

    def _process(self, inputs):
        raise NotImplementedError('   business logic is intentionally unimplemented')

    def _postprocess(self):
        raise NotImplementedError('   business logic is intentionally unimplemented')

url = "https://api.covid19india.org/data.json"
producer = DataProducer(url)
processor = DataProcessor()
visualizer = DataVisualizer()

graph = WorkflowGraph()
graph.connect(producer, "output", processor, "input")
graph.connect(processor, "output", visualizer, "input")
