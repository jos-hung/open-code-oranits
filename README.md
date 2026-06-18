
# ITS Simulation: DRL & Metaheuristic for Joint Task Handling

This project simulates joint task handling and mission processing in Intelligent Transportation Systems (ITS) using Deep Reinforcement Learning (DRL) and metaheuristic approaches.

---

## 1. Environment Setup

```
source setup.sh
```

## 2. Project Structure

```
.
├── src/
│   ├── DRL/                      # DRL algorithms & training scripts
│   ├── physic_definition/       # ITS environment simulation
│   ├── meta_heuristic/          # Metaheuristic methods & analysis
│   └── ...
├── configs/                     # Configuration files
├── task/                        # Output results (auto-created)
├── logs/                        # Runtime logs (auto-created)
├── run.py                       # Main entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 3. Run Simulations

Run simulations with `run.py`:

```bash
python run.py -i <method> [--verbose] [-device <cuda_id>] [-a <analysis_mode>] [-c <comparison_mode>]
```

### Examples:

- Run DDQN:

  ```bash
  python run.py -i ddqn
  ```

- Run many metaheuristics:

  ```bash
  python run.py -i many_metaheuristics
  ```

- Run evaluation of DDQN results:

  ```bash
  python run.py -i eval_ddqn
  ```

- Compare DRL and metaheuristics:

  ```bash
  python run.py -i meta_heuristic_proposal -c drl_and_meta_heuristic_proposal
  ```

- Run analysis mode 1:

  ```bash
  python run.py -i None -a 1
  ```

> `-a` triggers statistical analysis and plotting  
> `-device -1` uses CPU, `0` uses first CUDA GPU

---

## 4. Logging

All logs are automatically saved in the `logs/` folder, for example:

```
./logs/run_log_20250806_153245.log
```

---

## 5. Plotting Style

This project uses the `scienceplots` package for publication-ready matplotlib styles.

---

## 📚 Citation

If you use this codebase in your research, please cite the following works:

### 📄 1. arXiv Preprint

**Oranits: Mission Assignment and Task Offloading in Open RAN-based ITS using Metaheuristic and Deep Reinforcement Learning**  
Ngoc Hung Nguyen, Nguyen Van Thieu, Quang-Trung Luu, Anh Tuan Nguyen, Senura Wanasekara, Nguyen Cong Luong, Fatemeh Kavehmadavani, Van-Dinh Nguyen  
arXiv: [2507.19712](https://arxiv.org/abs/2507.19712)

```bibtex
@article{nguyen2025oranits,
  title     = {Oranits: Mission Assignment and Task Offloading in Open RAN-based ITS using Metaheuristic and Deep Reinforcement Learning},
  author    = {Nguyen, Ngoc Hung and Thieu, Nguyen Van and Luu, Quang-Trung and Nguyen, Anh Tuan and Wanasekara, Senura and Luong, Nguyen Cong and Kavehmadavani, Fatemeh and Nguyen, Van-Dinh},
  journal   = {arXiv preprint arXiv:2507.19712},
  year      = {2025},
  url       = {https://arxiv.org/abs/2507.19712}
}
```

---

### 📄 2. IEEE GLOBECOM 2025

**A Metaheuristic Approach for Mission Assignment and Task Offloading in Open RAN-Enabled Intelligent Transport Systems**  
Ngoc Hung Nguyen, Nguyen Van Thieu, Quang-Trung Luu, Vo Phi Son, Van-Dinh Nguyen  
Accepted at **IEEE GLOBECOM 2025**, Communication QoS, Reliability and Modeling Symposium

```bibtex
@inproceedings{nguyen2025metaheuristic,
  title     = {A Metaheuristic Approach for Mission Assignment and Task Offloading in Open RAN-Enabled Intelligent Transport Systems},
  author    = {Nguyen, Ngoc Hung and Nguyen, Van Thieu and Luu, Quang-Trung and Vo, Phi Son and Nguyen, Van-Dinh},
  booktitle = {Proceedings of the IEEE Global Communications Conference (GLOBECOM)},
  year      = {2025},
  organization = {IEEE}
}
```

---

📌 *Please cite one of the arXiv preprints or the GLOBECOM paper when using this code.*
