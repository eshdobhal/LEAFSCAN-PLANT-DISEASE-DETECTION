# LeafScan Surveillance: final presentation, speaker notes

*Esh Dobhal. Slides: [LeafScan_final_presentation.pdf](LeafScan_final_presentation.pdf) (19 slides, 16:9).*

## 1. LeafScan Surveillance

Welcome. This is LeafScan Surveillance, my Project 05 system. One phone photo, or a scouting video, goes in. Out comes the likely disease with an honest confidence, boxes around diseased leaves and pest damage, the share of tissue that is damaged, and a seven-day forecast. Everything I show is measured on held-out test data, including the parts that failed.

## 2. A scout has four questions, often days before an expert can come

Late or wrong diagnosis means spraying at the wrong time, with the wrong product, or not at all. The system maps one model to each question: a classifier, detectors, a segmenter and a forecaster.

## 3. Fourteen acceptance tests, written before any model was trained

Eight of the fourteen tests are shown. The others are lab recall, robustness, memory, explainability, safe use and reproducibility. Because the targets were fixed first, the go/no-go at the end could not be tuned to the results.

## 4. Six datasets, every split leakage-safe

Two honest gaps: no openly licensed annotated field video exists, so the video mode is shown on a clip assembled from photos and labelled synthetic; and no dataset follows the same plants over a season, so progression data are simulated from an established epidemiological model.

## 5. An MLP built from scratch, then taken apart by ablation

Everything here is NumPy: tensors, broadcasting, SVD, McCulloch-Pitts units, sigmoid neurons, an MLP with backprop, and the four optimisers. Biggest lessons: zero initialisation never breaks symmetry; BatchNorm gives the largest single gain; and more data shrinks overfitting more than any regulariser. Parameters and hyperparameters are logged separately for every run.

## 6. Five classic learning rules, run on real leaf photos

The pairs teach something: plain Hebbian learning diverges, and Oja’s decay term turns the same rule into principal component analysis. Unsupervised competitive learning finds real structure, but not the structure we care about, which is why the product uses supervised models.

## 7. One photo in, one report out

Each component has its own trained model and can be switched off. The web app and the browser version carry the classifier for growers; the CLI carries everything for scouts. The browser version runs the ONNX-exported ensemble on the phone and agrees with PyTorch on 219 of 225 field photos.

## 8. On field photos, every model lands around 70%

Same recipe for all three, same splits, test-time mirror augmentation. Lab accuracy is saturated; the meaningful number is field accuracy, where published PlantVillage-only models fall to 11–41%. Training with 30% field photos is what gets us to about 71%.

## 9. Faster R-CNN wins on PlantDoc; all three tie on tea

Faster R-CNN stands for the R-CNN family: it keeps R-CNN’s two-stage idea but replaces the slow selective-search proposals with a learned proposal network. Its proposal stage helps on PlantDoc’s cluttered web photos with many small similar leaves. Had the brief not fixed the detector family, the evidence here favours Faster R-CNN for disease detection.

## 10. Area is reliable; counting lesions is not

Severity is lesion pixels divided by leaf pixels, graded 0 to 5 at 1, 5, 10, 25 and 50 percent. Grades land within one step for 87.5% of test photos, which is what a scout needs. Lesion counts are only approximate, so the report leads with area and grade.

## 11. Part of the evidence is the backdrop

This is why the product tells users to photograph against a plain background, and why field photos make up 30% of training draws. It is also listed as the first bias in the model card.

## 12. An LSTM warns before the action threshold

From the last 14 days of severity, weather and spray records, the model predicts the next seven days. The LSTM has lower error and fewer false alarms; the RNN catches a few more crossings. The gates are sigmoid and the candidate is tanh, which is where step, sigmoid and tanh come together. The big caveat: no real season-long series was available, so this proves the method, not the field accuracy.

## 13. Synthetic leaves did not beat repeating the real ones

Potato late blight was cut to 25 training photos while seven classes kept 250. Conditional VAE and conditional DCGAN were trained only on that imbalanced set. Their small gains are within one standard deviation, while simple oversampling adds 15 points of recall. The latent space still clusters by class although the VAE is conditioned on it. Generative augmentation needs more data per class, or a generator pre-trained on many leaves, before it pays off.

## 14. of photos of other crops got a false pest alarm, so pest detection now runs only for tea

This is the most useful finding of the project. A metric on the in-domain test set (pest AP 0.96) said everything was fine; testing out of domain showed it would mislead users on every other crop. The fix is in code, not just in the documentation: the pipeline skips the pest model unless the crop is declared as tea, and skips the disease classifier for tea, which it never learned.

## 15. Calibrated, robust and fast on a CPU

Every number comes from surveillance/evaluate_all.py: macro F1, mAP, IoU, ECE, latency, memory and 16 corruptions, on test sets used once. The package also caught two bugs of its own, both now covered by tests: inference-mode tensors refused by autograd, and a temperature that could go negative.

## 16. 12 of 14 acceptance tests pass

Both failures are about lesion-level precision. Damaged area and grade are reliable to within one grade for 87.5% of photos, so reports lead with those and show counts as approximate. Conditions: pests on tea only (enforced in code), the forecaster stays a guide until validated on real data, and re-measure on a local test set before use in a new region.

## 17. Pilot first, with an extension officer in the loop

Infrastructure is small: a 1 GB instance runs the web app; a 4-vCPU, 4 GB worker runs the full pipeline at about 2,600 photos an hour. Every model version is a GitHub release with checksums, so rolling back is a one-line change. The biggest risk is a wrong diagnosis leading to a wrong spray, which is why a person reviews reports through the pilot.

## 18. It advises; a person decides

Known biases: mostly North American lab photos and Asian tea photos, few field photos, and a classifier that partly reads backdrops. The model card lists each one with the evidence.

## 19. Where the next gains are

Thank you. The technical report, model card, dataset card, user guide and demo video are all in the repository, and every number in this deck can be reproduced with the commands in the report.
