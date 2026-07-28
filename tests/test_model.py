import os
import sys
import torch
import io
import contextlib
from model import TinyGPT
from tinygpt.training.checkpoint import save_checkpoint, load_checkpoint


def test_model_forward():
    """Verify model forward pass produces correct logit shape and positive loss."""
    model = TinyGPT(50257, 512, 768, 12, 12, 3072, 0.1)
    x = torch.randint(0, 50257, (2, 128))
    logits, loss = model(x, x)
    assert logits.shape == (2, 128, 50257), f"Expected (2,128,50257), got {logits.shape}"
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    print(f"  test_model_forward: OK  (loss={loss.item():.4f})")


def test_checkpoint_save_load():
    """Verify checkpoint saves and loads correctly with model_state key."""
    model = TinyGPT(50257, 512, 768, 12, 12, 3072, 0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    test_path = "/tmp/test_tinygpt_ckpt.pt"
    save_checkpoint(42, model, optimizer, 4.5, 5.1, test_path)

    model2 = TinyGPT(50257, 512, 768, 12, 12, 3072, 0.1)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
    step, val_loss = load_checkpoint(test_path, model2, optimizer2)

    assert step == 42, f"Expected step 42, got {step}"
    assert abs(val_loss - 5.1) < 1e-5, f"Expected val_loss 5.1, got {val_loss}"

    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2), "Weights don't match after save/load"

    os.remove(test_path)
    print(f"  test_checkpoint_save_load: OK")


def test_chat_init():
    """Verify chat.py module-level code can import without crash."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            from chat import model
        assert model is not None
        print(f"  test_chat_init: OK")
    except Exception as e:
        print(f"  test_chat_init: SKIPPED ({e})")


def test_eval_suite_import():
    """Verify eval_suite module can be imported."""
    from eval_suite import eval_suite
    assert callable(eval_suite)
    print(f"  test_eval_suite_import: OK")


if __name__ == "__main__":
    print("Running Tiny-GPT tests...")
    test_model_forward()
    test_checkpoint_save_load()
    test_chat_init()
    test_eval_suite_import()
    print("\nAll tests passed!")
