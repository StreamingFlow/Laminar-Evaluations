from dispel4py.base import (
    ConsumerPE,
    IterativePE,
)
from dispel4py.workflow_graph import WorkflowGraph


class StreamRead(IterativePE):

    def __init__(self):
        IterativePE.__init__(self)

    def _process(self, filename):
        raise NotImplementedError('   business logic is intentionally unimplemented')



class StreamToFile(ConsumerPE):

    def __init__(self):
        ConsumerPE.__init__(self)

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class Decimate(IterativePE):

    def __init__(self):
        IterativePE.__init__(self)

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class Detrend(IterativePE):

    def __init__(self):
        IterativePE.__init__(self)

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class Demean(IterativePE):

    def __init__(self):
        IterativePE.__init__(self)

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class RemoveResponse(IterativePE):

    def __init__(self):
        IterativePE.__init__(self)

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class Filter(IterativePE):

    def __init__(self):
        IterativePE.__init__(self)

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class CalculateNorm(IterativePE):

    def __init__(self):
        IterativePE.__init__(self)

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class Whiten(IterativePE):

    def __init__(self):
        IterativePE.__init__(self)

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class CalculateFft(IterativePE):

    def __init__(self):
        IterativePE.__init__(self)

    def _process(self, data):
        raise NotImplementedError('   business logic is intentionally unimplemented')


graph = WorkflowGraph()

streamRead = StreamRead()
streamToFile = StreamToFile()
decim = Decimate()
detrend = Detrend()
demean = Demean()
removeResponse = RemoveResponse()
filt = Filter()
calNorm = CalculateNorm()
whiten = Whiten()
calcFft = CalculateFft()

graph.connect(streamRead, 'output', decim, 'input')
graph.connect(decim, 'output', detrend, 'input')
graph.connect(detrend, 'output', demean, 'input')
graph.connect(demean, 'output', removeResponse, 'input')
graph.connect(removeResponse, 'output', filt, 'input')
graph.connect(filt, 'output', calNorm, 'input')
graph.connect(calNorm, 'output', whiten, 'input')
graph.connect(whiten, 'output', calcFft, 'input')
graph.connect(calcFft, 'output', streamToFile, 'input')
