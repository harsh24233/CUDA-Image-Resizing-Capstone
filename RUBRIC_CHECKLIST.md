# Capstone Rubric Checklist (implementation package)

This checklist maps the repository to the supplied Coursera rubric. It does not claim that environment-dependent GPU execution has already been completed.

## 1. Code repository (40 points)
- [x] Repository-ready project structure and separate source/header files.
- [x] Descriptive README with build/run commands, requirements, CLI, limitations, and troubleshooting.
- [x] Command-line input path, output path, and scale factor.
- [x] Makefile and single-run helper script.
- [x] Clear function responsibilities, error messages, and comments for non-obvious behavior.
- [x] Validation-suite and large-input generator scripts.
- [ ] Publish this folder to a public GitHub/GitLab/Bitbucket repository and add its URL to the course submission.
- [ ] Compile and run on a real CUDA/NPP environment; fix any version-specific build/API issues exposed there.

## 2. Proof of execution (20 points)
- [x] Scripts are provided to capture environment/build logs, multiple real run logs, output PGM files, output validation, and measured timing CSV.
- [x] Optional 100-megapixel (approximately 100 MB pixel payload) stress input generation is included.
- [ ] Run `bash scripts/run_validation_suite.sh --large` on a suitable GPU environment.
- [ ] Create `submission_artifacts/proof_of_execution.zip` with `bash scripts/package_proof.sh`.
- [ ] Inspect the ZIP and make sure the logs show `Status: SUCCESS`, the output images exist, and timings are present.

No execution logs or benchmark values are fabricated in this package. Until the unchecked steps are completed, this rubric section is not yet submission-ready.

## 3. Project description (20 points)
- [x] `PROJECT_DESCRIPTION.md` explains the goal, interface, processing pipeline, design choices, timing scope, challenges, lessons, and future work.
- [ ] After testing, replace/extend the validation section with observations supported by the actual logs and measurements.

## 4. Presentation/demonstration (20 points)
- Not included in this package, per the current scope. The course rubric separately asks for a 5–10 minute recorded presentation URL.
