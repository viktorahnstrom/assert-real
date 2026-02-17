# XADE Documentation Checklist

## Completed
- [X] API Contract (`/docs/api-contract.md`)
- [ ] System Architecture (`/docs/api-architecture.md`)

## Required Documents

### Technial Documentation

- [ ] **Database Schema** (`docs/database-schema.md`)
    - Tables: analyses, images, users (future)
    - Relationships and indexes
    - Migratation stratergy

- [ ] **ML Model Documentation** (`docs/ml-models.md`)
    - Detection models (CLIP, EfficientNet)
    - VLM providers comparison
    - Model selection criteria
    - Performance benchmarks

- [ ] **Deployment Guide** (`docs/deployment.md`)
    - Local development setup
    - Production deployment (Docker, cloud services)
    - Environment variables
    - Security considerations

## Maybe need:

- [ ] **Testing Strategy** (`docs/testing-strategy.md`)
    - Unit tests (backend, detection, VLM)
    - Integration tests (API endpoints)
    - E2E tests (desktop/mobile apps)
    - User testing protocol

---

### User-Facing Documentation

- [ ] **User Guide - Desktop** (`docs/user-guide-desktop.md`)
    - Installation instructions
    - Feature walkthrough
    - Troubleshooting

- [ ] **User Guide - Mobile** (`docs/user-guide-mobile.md`)
    - Installation (iOS/Android)
    - Camera/gallery usage
    - Intepreting results

- [ ] **Developer Guide** (`docs/developer-guide.md`)
    - Codebase structure
    - Adding new VLM providers
    - Contributing guidelines

---

### Research Documentation

- [ ] **Literature Review Summary** (`docs/literature-review.md`)
    - Key papers analyzed
    - Gaps identified
    - Related work comparison

- [ ] **Dataset Documentation** (`docs/datasets.md`)
    - FaceForensics++, Celeb-DF, DFDC, DF-40
    - Preprocessing steps
    - Evaluation protocols

- [ ] **User Study Protocol** (`docs/user-study-protocol.md`)
    - Participant recruitment
    - Study taks
    - Questionnaire design (Likert scales, open-ended)
    - Ethical approval documentation

- [ ] **Evaluation Metrics** (`docs/evaluation-metrics.md`)
    - Technical: AUC, precision, recall, F1
    - User: comprehnsion scores, trust calibration
    - Explanation quality metrics

---

### Project Management
- [ ] **Risk Assessment** (`docs/risk-assessment.md`)
    - Technical risks
    - Timeline risks
    - Mitigation stratergies

- [ ] **Weekly Progress Reports** (`docs/progress/`)
  - Week-1.md, Week-2.md, etc.
  - What was done, blockers, next steps
