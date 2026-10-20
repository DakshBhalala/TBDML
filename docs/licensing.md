# Licensing and Commercial Use Audit

*Last Reviewed: October 2026*  
*Project: CompactTab*  
*Objective: Rigorous verification of commercial usability, open-source compliance, and dependency governance.*

---

## 1. Governance Principles

CompactTab is developed with the long-term objective of unrestricted commercial viability in enterprise machine learning deployments. To ensure complete intellectual property and legal compliance:
1. **Source Code vs. Model Weights Distinction:** The licensing of source code, model weights, and training datasets must be tracked independently.
2. **Copyleft Avoidance:** Strong copyleft licenses (GPL, AGPL) are prohibited in production dependencies to protect proprietary integration. Permissive licenses (Apache-2.0, MIT, BSD-2/3-Clause) are favored.
3. **Model Weight Provenance:** Models pre-trained on datasets or synthetically generated under non-commercial or ambiguous terms (e.g., CC-BY-NC, Prior Labs restricted weights) are strictly excluded from commercial inference artifacts.
4. **No Unverified Assumptions:** Public availability does not imply commercial permission. Any ambiguous asset is flagged as `NEEDS REVIEW`.

---

## 2. Component and Dependency Audit Table

| Component / Asset | Type | Source URL / Repository | Stated License | Commercial Use Permitted? | Restrictions / Attribution Requirements | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Python** | Runtime | python.org | PSF License | Yes | Standard PSF attribution in notices. | **Verified** |
| **PyTorch** | Deep Learning Framework | github.com/pytorch/pytorch | BSD-3-Clause | Yes | Include copyright notice and disclaimer. | **Verified** |
| **scikit-learn** | ML Library | github.com/scikit-learn/scikit-learn | BSD-3-Clause | Yes | Include copyright notice and disclaimer. | **Verified** |
| **LightGBM** | GBDT Engine | github.com/microsoft/LightGBM | MIT | Yes | Standard MIT license text retention. | **Verified** |
| **XGBoost** | GBDT Engine | github.com/dmlc/xgboost | Apache-2.0 | Yes | Apache-2.0 notice retention and patent grant. | **Verified** |
| **CatBoost** | GBDT Engine | github.com/catboost/catboost | Apache-2.0 | Yes | Apache-2.0 notice retention. | **Verified** |
| **NumPy** | Array Library | github.com/numpy/numpy | BSD-3-Clause | Yes | Standard copyright notice retention. | **Verified** |
| **Pandas** | DataFrames | github.com/pandas-dev/pandas | BSD-3-Clause | Yes | Standard copyright notice retention. | **Verified** |
| **SciPy** | Scientific Library | github.com/scipy/scipy | BSD-3-Clause | Yes | Standard copyright notice retention. | **Verified** |
| **Typer** | CLI Framework | github.com/tiangolo/typer | MIT | Yes | Standard MIT license text retention. | **Verified** |
| **Rich** | Terminal Output | github.com/Textualize/rich | MIT | Yes | Standard MIT license text retention. | **Verified** |
| **Pydantic** | Schema Validation | github.com/pydantic/pydantic | MIT | Yes | Standard MIT license text retention. | **Verified** |
| **Psutil** | Resource Profiling | github.com/giampaolo/psutil | BSD-3-Clause | Yes | Standard copyright notice retention. | **Verified** |
| **TabArena Benchmark** | Benchmark Code | github.com/autogluon/tabarena | Apache-2.0 | Yes | Code is Apache-2.0; individual benchmark datasets carry separate licenses. | **Verified** |
| **TabM Reference Code** | Neural Reference | github.com/yandex-research/tabm | Apache-2.0 | Yes | Implementation code is Apache-2.0. | **Verified** |
| **TabPFN Source Code** | Model Code | github.com/prior-labs/TabPFN | Apache-2.0 | Yes | Code is open source. | **Verified** |
| **TabPFN v2 / v3 Pretrained Weights** | Model Weights | HuggingFace / Prior Labs | Prior Labs Custom / Non-commercial | **NO** (Without custom enterprise contract) | Pretrained weights prohibit unauthorized commercial deployment; requires API/token authorization. | **RESTRICTED** (Do NOT bundle weights) |

---

## 3. Dataset Licensing Audit (Benchmarking & Evaluation)

| Dataset | Modality / Domain | Primary Source | Documented License | Commercial Use Permitted? | Notes / Usage Boundary |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **California Housing** | Regression (Continuous) | US Census Bureau (1990) / StatLib | Public Domain | Yes | Standard regression benchmark. No PII. |
| **Breast Cancer Wisconsin** | Binary Classification | UCI ML Repository / Wolberg et al. | CC-BY 4.0 | Yes (with attribution) | Standard clinical benchmark. Strictly de-identified. |
| **Wine Recognition** | Multiclass Classification | UCI ML Repository / Forina et al. | CC-BY 4.0 | Yes (with attribution) | Chemical analysis benchmark (3 classes, 13 features). |
| **Diabetes** | Regression (Continuous) | Efron et al. (2004) / LARS paper | Public Domain | Yes | 10 baseline variables, 442 instances. |
| **OpenML-CC18 Datasets** | Benchmark Suite | openml.org | Varies per dataset (CC-BY, CC0, etc.) | Requires per-task check | Benchmark only; must verify license before proprietary training. |

---

## 4. Operational Guidelines for Future Development

1. **In-house Training Exclusivity:** CompactTab models developed herein are trained exclusively on customer or open-licensed verified data (or synthetically generated in-house using unrestricted algorithms).
2. **Third-Party Pretrained Checkpoints:** No pretrained checkpoints from restrictive sources (e.g. TabPFN Prior Labs weights) may be incorporated into commercial shipping builds without prior legal review.
3. **Artifact Retention:** Whenever datasets or weights are downloaded for benchmarking, the originating URL, license text, and download timestamp must be preserved in `artifacts/metadata/`.
