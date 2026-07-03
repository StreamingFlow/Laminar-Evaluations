from dispel4py.core import GenericPE
from dispel4py.base import IterativePE
from dispel4py.workflow_graph import WorkflowGraph

class ReadRaDec(GenericPE):
    def __init__(self):
        GenericPE.__init__(self)
        self._add_output('output')

    def _process(self, inputs):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class GetVOTable(IterativePE):
    def __init__(self):
        IterativePE.__init__(self)

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class FilterColumns(IterativePE):
    def __init__(self):
        IterativePE.__init__(self)
        self.columns = ['MType', 'logR25']

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class InternalExtinction(IterativePE):

    def __init__(self):
        IterativePE.__init__(self)

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


graph = WorkflowGraph()
readData = ReadRaDec()
votab = GetVOTable()
filt = FilterColumns()
intext = InternalExtinction()
graph.connect(readData, 'output', votab, 'input')
graph.connect(votab, 'output', filt, 'input')
graph.connect(filt, 'output', intext, 'input')
