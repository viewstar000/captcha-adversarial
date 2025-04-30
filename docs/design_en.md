Design Philosophy
===================

[中文](design.md) | [English](design_en.md)

Copyright (c) 2025 viewstar000

This is a human-machine CAPTCHA experimental project based on the adversarial design approach.

The main design philosophy is divided into four aspects:

- Semantic Countermeasure
- Behavioral Countermeasure
- Frontend Countermeasure
- Backend Countermeasure

Semantic Countermeasure
-------------------

The goal of semantic countermeasure is to design puzzles that are friendly to humans but difficult for machines, essentially a type of image-based test. However, with the rapid development of deep learning and large language models in recent years, we can no longer rely on simple semantic countermeasures to effectively defend against machine attacks. On the other hand, any semantic countermeasure cannot completely avoid human user misunderstandings and incorrect inputs. Therefore, in scenarios requiring optimal user experience, many CAPTCHA designs have abandoned semantic countermeasures entirely, relying only on other countermeasure methods.

But semantic countermeasures still have their significance. From past experiences, even the simplest semantic countermeasures can block over 90% of attack traffic. However, we need to set more practical goals for semantic countermeasures, such as increasing the technical difficulty and operational cost of machine cracking while ensuring human user experience. At minimum, it should not be easily cracked through simple brute-force guessing, text matching, or image recognition techniques.

To achieve these goals, we can consider the following aspects:

1. The state space is sufficiently large to prevent attackers from easily cracking CAPTCHAs through random guessing.
2. There are enough materials available, and they can be updated at a low cost to avoid being cracked by simple text/image matching techniques.
3. Use adversarial attack samples to distort materials, reducing the success rate of cracking CAPTCHAs through deep learning or AI models.

Behavioral Countermeasure
-------------------

Behavioral countermeasures involve modeling user mouse operations, keyboard inputs, touch screen interactions, etc., to identify whether requests are from humans or machines.

The simplest way for attackers to bypass behavioral detection is to use recording and replay attacks, directly replaying behavior sequences that can pass detection, or adding random perturbations to these sequences to bypass frequency control detection. To effectively identify such attack methods, we need to fully utilize the information asymmetry advantage of defenders. Typically, defenders can leverage their business scale advantages to collect more comprehensive user behavior data, while attackers usually only collect limited real user data. Therefore, the behavior sequence data forged by attackers will inevitably show obvious differences in statistical distribution compared to real user data.

Based on this characteristic, we can design two types of models for behavioral countermeasures:

- Basic model: Used to model genuine user behaviors, logically a single-classification problem. Practically, it can be trained using real user data + noise data for binary classification. This model focuses on accuracy (user experience) and has low requirements for update timeliness.
- Adversarial model: Used to identify specific attack batches' malicious behaviors. It can use offline mining or anomaly detection system outputs of attack samples + genuine user white samples for binary classification training. It requires high update timeliness. In practice, simple models like decision trees can achieve good blocking effects.

Frontend Countermeasure
-------------------

The goal of frontend countermeasures is to prevent attackers from directly launching HTTP requests to business interfaces without relying on browser/client environments. Its significance lies in the huge difference in resource consumption and attack QPS between two attack modes. As long as attackers cannot operate outside the browser/client environment, the overall scale of each attack can be effectively controlled. Additionally, since attackers cannot operate outside the browser/client environment, defenders can more easily locate the source of attacks using browser fingerprints, device fingerprints, etc., enabling more targeted traffic control.

Common frontend countermeasures include browser fingerprints, device fingerprints, parameter dynamic signature verification, PoW, emulator detection, debugger detection, code dynamic obfuscation, etc.

Backend Countermeasure
-------------------

Utilize big data analysis, data mining, machine learning, and other methods to mine and identify attack traffic. The core lies in designing and implementing anomaly detection algorithms, requiring corresponding detection algorithms, visualization analysis tools, and strategy countermeasure frameworks for traffic statistical features, sequence features, group characteristics, etc.
