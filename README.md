# Spectrum-Informed PINN for 2D Helmholtz / 频谱感知 PINN 求解二维亥姆霍兹方程

> Three-stage progressive Physics-Informed Neural Network (PINN) in JAX for solving
> the 2D Helmholtz equation in electromagnetic scattering (cylindrical obstacle in
> rectangular domain). Uses Spectrum-Informed Multistage Neural Network (SI-MSNN)
> initialization — FFT residual analysis → main-frequency weight init → residual fusion.
>
> 基于 JAX 的三阶段渐进式物理信息神经网络（PINN），求解电磁散射场景下的二维亥姆霍兹方程
>（矩形区域内的圆柱散射体）。采用频谱感知多阶段神经网络（SI-MSNN）初始化：
> 残差 FFT 分析 → 主频权重初始化 → 残差式融合。

---

## Highlights / 项目亮点

- **SI-MSNN initialization / 频谱感知初始化**：FFT the stage residual, extract top-n
  frequency components, and directly initialize the first-layer weights — explicit
  high-frequency guidance instead of blind Xavier.
- **Three-stage progressive training / 三阶段渐进训练**：low-freq → mid-freq → high-freq,
  each stage fuses onto the previous via residual addition (`u_prev + net(z)`).
- **Full physics constraints / 完整物理约束**：Dirichlet/Neumann boundary conditions,
  PDE residual loss, dynamic medium switching at cylinder interface.
- **JAX autodiff + hybrid optimizer / JAX 自动微分 + 混合优化**：`vjp`/`vmap` for
  efficient gradient computation; Adam (fast descent) → L-BFGS (fine-tuning).
- **PyTorch baseline included / 含 PyTorch 基线**：`examples/pytorch_helmholtz.py` —
  simpler PINN implementation for comparison.

## Architecture / 训练流程

```
Stage 1: Standard PINN (Xavier init)
    → u_pred1, capture low-frequency structure
    ↓  residual = u_true - u_pred1
Stage 2: FFT(residual) → top-n frequencies → init layer-1 weights
    → u_pred2 = u_pred1 + net2(z)
    ↓  residual = u_true - u_pred2
Stage 3: FFT(residual) → new top-n frequencies → init layer-3 weights
    → u_pred3 = u_pred2 + net3(z)   (final high-precision solution)
```

Each stage optimizes with `loss = w_pde * PDE_residual + w_bc * boundary_loss`,
using Adam then L-BFGS.

## Quick start / 快速开始

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python helmholtz_pinn.py
```

The script trains all three stages, prints error statistics, and saves prediction /
exact-solution / error-distribution plots to the working directory.

### PyTorch baseline

```bash
python examples/pytorch_helmholtz.py
```

## Key parameters / 主要参数

| Parameter | Effect |
|---|---|
| `n_hl`, `n_unit` | Network depth & width |
| `N_col` | Collocation points per stage |
| `n_f`, `n_f2` | Number of dominant frequencies for SI init |
| `epoch1/2/3` | Training epochs per stage |
| `lw` | Loss weights `[pde_residual, boundary]` |
| `boundary_num` | Boundary sample count |

## Not included / 未包含项

| Excluded | Reason |
|---|---|
| Reference PDFs (~44 MB) | Academic papers (SI-MSNN, MscaleDNN, etc.) |
| `output3/`, plot PNGs | Generated result images (reproduced by running) |
| `ODE2d.zip` | Archive duplicate |
| `output3/ODE2d_Helmholtz3.py` | Earlier variant superseded by `helmholtz_pinn.py` |

## Tech stack / 技术栈

`Python 3.8+` · `JAX` · `optax` · `TensorFlow Probability (JAX)` · `pyDOE` · `SciPy` · `matplotlib`

## License

[MIT](LICENSE)
