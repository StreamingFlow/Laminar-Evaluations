from dispel4py.base import ProducerPE, IterativePE, ConsumerPE
from dispel4py.workflow_graph import WorkflowGraph


class NumberProducer(ProducerPE):

    def __init__(self):
        ProducerPE.__init__(self)

    def _process(self, inputs):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class IsPrime(IterativePE):

    def __init__(self):
        IterativePE.__init__(self)

    def _process(self, num):
        raise NotImplementedError('   business logic is intentionally unimplemented')


class PrintPrime(ConsumerPE):
    def __init__(self):
        ConsumerPE.__init__(self)

    def _process(self, num):
        raise NotImplementedError('   business logic is intentionally unimplemented')


graph = WorkflowGraph()

producer = NumberProducer()
isprime = IsPrime()
printprime = PrintPrime()

graph.connect(producer, 'output', isprime, 'input')
graph.connect(isprime, 'output', printprime, 'input')
