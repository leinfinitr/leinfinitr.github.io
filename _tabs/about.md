---
# the default layout is 'page'
icon: fas fa-info-circle
order: 4
---

Hi, I'm **Jialong Liu** (刘佳隆), an M.S. student in Software Engineering at
[Shanghai Jiao Tong University](https://www.sjtu.edu.cn/), advised by
[Prof. Rong Chen](https://ipads.se.sjtu.edu.cn/rong_chen) in the
[Institute of Parallel and Distributed Systems (IPADS)](https://ipads.se.sjtu.edu.cn/).
I expect to graduate in March 2028.

My interests lie at the intersection of **systems and AI infrastructure**.I enjoy
turning systems ideas into working software and evaluating them under realistic
resource constraints. I am always happy to discuss systems research, open-source 
collaboration, and engineering opportunities.

## Research Interests

- Operating systems and runtime systems
- Scheduling for GPUs, NPUs, and heterogeneous accelerators
- AI inference systems and model serving
- Memory management and resource multiplexing
- High-performance and latency-sensitive systems

## Selected Work

### XSched: Preemptive Scheduling for Diverse XPUs

[XSched](https://github.com/XpuOS/xsched) is a scheduling framework that brings
flexible preemptive scheduling to diverse accelerators, including GPUs, NPUs,
ASICs, and FPGAs. My contributions include implementing Intel Level Zero
support for Intel iGPUs and NPUs, porting XSched to Windows, and building its
cross-platform IPC library.

The work appeared at the **OSDI '25**:

> Weihang Shen, Mingcong Han, **Jialong Liu**, Rong Chen, and Haibo Chen.<br>
> “XSched: Preemptive Scheduling for Diverse XPUs.”<br>
> [Paper](https://www.usenix.org/conference/osdi25/presentation/shen-weihang)
> · [Code](https://github.com/XpuOS/xsched)

### vLLM Switch

I am developing mechanisms for faster model switching in vLLM by extending
sleep mode with memory- and disk-backed model-weight snapshots. I also designed
and implemented the
[vLLM Switch Controller](https://github.com/leinfinitr/vllm-switch-controller),
an external control plane for request-driven switching and co-location across
long-lived vLLM backends.

### XServer

I built a dynamic memory scheduler on top of a memory-adjustable inference
engine, enabling concurrent AI workloads to operate under tight memory budgets.
This project explores memory-aware resource multiplexing for heterogeneous AI
platforms.

### OhDiff

[OhDiff](https://github.com/PenXLa/oh-diff) is a high-performance differential
backup system. I led its early systems design and designed both simulation-based
and real-machine evaluations. The project received **First Prize** in the 2024
China Graduate Operating System Open Source Innovation Competition.

## Education & Teaching

I received my B.E. in Software Engineering from Shanghai Jiao Tong University
in 2025 and am continuing there as an M.S. student. 

From September 2025 to June
2026, I served as a teaching assistant for *Introduction to Computer Systems*,
supporting a class of 95 undergraduate students. I received SJTU's
[Excellent Teaching Assistant Award](https://ctld.sjtu.edu.cn/news/detail/1328)
for the Spring 2026 semester.
