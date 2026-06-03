import time
from datetime import datetime


class AgentMonitor:

    def __init__(self):
        self.logs = []

    def start_trace(self, agent_name):

        trace = {
            "agent_name": agent_name,
            "start_time": time.time(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return trace

    def end_trace(
        self,
        trace,
        tokens_used=0,
        success=True,
        tool_calls=0
    ):

        end_time = time.time()

        trace["end_time"] = end_time

        trace["latency_seconds"] = round(
            end_time - trace["start_time"],
            2
        )

        trace["tokens_used"] = tokens_used

        trace["estimated_cost_usd"] = round(
            (tokens_used / 1000) * 0.002,
            4
        )

        trace["success"] = success

        trace["tool_calls"] = tool_calls

        self.logs.append(trace)

    def get_logs(self):
        return self.logs

    def print_logs(self):

        print("\n========== AGENT OBSERVABILITY ==========\n")

        for log in self.logs:

            print(f"Agent: {log['agent_name']}")
            print(f"Latency: {log['latency_seconds']} sec")
            print(f"Tokens: {log['tokens_used']}")
            print(f"Cost: ${log['estimated_cost_usd']}")
            print(f"Tool Calls: {log['tool_calls']}")
            print(f"Success: {log['success']}")
            print("-" * 40)