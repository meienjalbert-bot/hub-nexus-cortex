import asyncio


class SurvivalGate:
    def __init__(self):
        self.lock = asyncio.Lock()


survival_gate = SurvivalGate()
