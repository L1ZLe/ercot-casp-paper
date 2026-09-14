# Framework Diagram Prompt

**Paper**: SPARC: Constraint-Aware Spread Predictor for ERCOT Day-Ahead LMP Spreads

## Image Generation Prompt

Create a clean academic architecture diagram for a neural forecasting model titled "SPARC: Constraint-Aware Spread Predictor." The illustration flows left-to-right across three distinct processing stages, rendered in a flat vector-art style on a white background (#FFFFFF) with subtle drop shadows on all modules.

**Color palette:** Primary modules in muted blue #4477AA, learned embeddings and attention components in teal #44AA99, temporal features in soft purple #AA3377, target and loss components in warm accent #CCBB44, arrows and connectors in dark grey #555555.

**Stage 1 – Input Features (left):** Three vertically stacked input blocks. Top block labeled "Constraint Slots C_t" shows K=50 rounded rectangles (muted blue) each containing smaller labeled sub-fields: "μ_k" (shadow price), "ID_k" (constraint index), "v_k" (kV level), "f_k" (flow ratio). A small note beneath reads "Top-K by |μ|, zero-padded." Middle block labeled "Pair Embeddings" shows two teal circles side-by-side, labeled "src: HB_HUBAVG" and "snk: HB_PAN," each outputting an 8-dim vector. Bottom block labeled "Temporal Features u_t" shows three purple rounded boxes containing "y_{t-24}", "y_{t-48}", "y_{t-168}" and a Fourier symbol with "18-dim" annotation.

**Stage 2 – Constraint-Aware Attention (center):** This is the core module, a large rounded rectangle in muted blue with teal internal components. Inside, a horizontal flow: a "Pair Query q_t" projection teal box feeds into a "Scaled Dot-Product Attention" block (softmax symbol) receiving "Keys h_k" from a shared MLP encoder applied to each constraint slot. A parallel branch labeled "Latent μ_k" (teal) applies a projection ψ_μ to raw shadow prices. The attention weights a_k multiply the latent μ_k values. A large summation symbol Σ_k computes the latent spread. A prominent annotation reads "ΔSF_k = a_k" with the equation "spread = Σ ΔSF_k · μ_k" beneath. A dashed box encloses this attention mechanism labeled "Energy-Cancel Property: λ_src - λ_snk cancels."

**Stage 3 – Quantile Projection (right):** The latent spread, pair embeddings, and temporal features converge via three input arrows into a "Concatenate" node, feeding a "Quantile Head" module (muted blue). This module outputs seven vertically arranged lines labeled with quantiles: "Q10, Q25, Q45, Q50, Q55, Q75, Q90." A soft constraint icon between quantile lines indicates the "Non-Crossing Penalty (λ=0.1)." Below, a warm accent #CCBB44 box labeled "Average Quantile Loss" receives all quantile outputs and the target spread y_t.

**Connections:** Bold directional arrows (dark grey #555555) connect all stages. A feedback arrow loops from the output back to temporal features labeled "Lag-24/48/168." Include small legend in bottom-right corner with color swatches and labels: "Constraint Features," "Learned Embeddings," "Temporal," "Loss/Target."

## Usage Instructions

1. Copy the prompt above into an AI image generator (DALL-E 3, Midjourney, Ideogram, etc.)
2. Generate the image at high resolution (2048x1024 or similar landscape)
3. Save as `framework_diagram.png` in the same `charts/` folder
4. Insert into the paper's Method section using:
   - LaTeX: `\includegraphics[width=\textwidth]{charts/framework_diagram.png}`
   - Markdown: `![Framework Overview](charts/framework_diagram.png)`
