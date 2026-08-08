"""Gradient Accumulation task."""

TASK = {
    "title": "Gradient Accumulation",
    "difficulty": "Easy",
    "function_name": "accumulated_step",
    "hint": "Zero grads once. For each micro-batch: forward, loss/n_batches, backward. Then optimizer.step(). The loss scaling ensures accumulated grads match a single large batch.",
    "tests": [
        {
            "name": "Matches full batch update",
            "code": "\nimport torch, torch.nn as nn\ntorch.manual_seed(0)\nmodel = nn.Linear(4, 2, bias=False)\nmodel_ref = nn.Linear(4, 2, bias=False)\nmodel_ref.load_state_dict(model.state_dict())\nloss_fn = nn.MSELoss()\nopt = torch.optim.SGD(model.parameters(), lr=0.1)\nopt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.1)\nx1, y1 = torch.randn(2, 4), torch.randn(2, 2)\nx2, y2 = torch.randn(2, 4), torch.randn(2, 2)\n{fn}(model, opt, loss_fn, [(x1, y1), (x2, y2)])\nopt_ref.zero_grad()\nloss_ref = loss_fn(model_ref(torch.cat([x1, x2])), torch.cat([y1, y2]))\nloss_ref.backward()\nopt_ref.step()\nassert torch.allclose(model.weight.data, model_ref.weight.data, atol=1e-5), 'Must match full batch'\n"
        },
        {
            "name": "Returns loss value",
            "code": "\nimport torch, torch.nn as nn\nmodel = nn.Linear(4, 2)\nopt = torch.optim.SGD(model.parameters(), lr=0.01)\nloss = {fn}(model, opt, nn.MSELoss(), [(torch.randn(2, 4), torch.randn(2, 2))])\nassert isinstance(loss, float), f'Should return float, got {type(loss)}'\nassert loss > 0, 'Loss should be positive'\n"
        },
        {
            "name": "Parameters actually update",
            "code": "\nimport torch, torch.nn as nn\nmodel = nn.Linear(4, 2)\nopt = torch.optim.SGD(model.parameters(), lr=0.1)\nw_before = model.weight.data.clone()\n{fn}(model, opt, nn.MSELoss(), [(torch.randn(2, 4), torch.randn(2, 2))])\nassert not torch.equal(model.weight.data, w_before), 'Should change'\n"
        }
    ]
}
