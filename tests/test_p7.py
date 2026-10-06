import os
import unittest
import torch
import psutil
import subprocess
from tinygpt.device import resolve_device
from tinygpt.training.checkpoint import convert_optimizer_state

class TestP7(unittest.TestCase):
    def test_optimizer_format_conversion(self):
        # Create a dummy model
        model = torch.nn.Linear(10, 10)
        params = list(model.parameters())
        
        # Fake offload state
        offload_state = {
            "t": 10,
            "master": [p.data.float().cpu() for p in params],
            "m": [torch.ones_like(p) for p in params],
            "v": [torch.ones_like(p) * 2 for p in params],
        }
        
        # Convert to torch
        torch_state = convert_optimizer_state(offload_state, "torch", params)
        self.assertIn("state", torch_state)
        self.assertIn("param_groups", torch_state)
        self.assertEqual(len(torch_state["state"]), len(params))
        
        # Convert back
        offload_back = convert_optimizer_state(torch_state, "offload", params)
        self.assertEqual(offload_back["t"], 10)
        self.assertTrue(torch.allclose(offload_back["m"][0], offload_state["m"][0]))
        self.assertTrue(torch.allclose(offload_back["v"][0], offload_state["v"][0]))
        
    def test_forced_cpu_device(self):
        os.environ["TINYGPT_DEVICE"] = "cpu"
        device = resolve_device()
        self.assertEqual(device, "cpu")
        
    def test_import_side_effects(self):
        # Importing main should not spike memory
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        
        import main
        import chat
        import eval_suite
        
        mem_after = process.memory_info().rss
        diff_mb = (mem_after - mem_before) / (1024 * 1024)
        
        # Memory increase should be minimal (just module loading, no large allocations)
        self.assertLess(diff_mb, 100, f"Imports caused memory spike of {diff_mb:.2f} MB")

if __name__ == "__main__":
    unittest.main()
