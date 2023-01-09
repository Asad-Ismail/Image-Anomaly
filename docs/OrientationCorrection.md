
## Orientation Correction

We devloped digital solution to correct the vegetable orientation, instead of manually changing the orientation of the vegetable or employing dedicated (and expensive) machinery to physically move the vegetables, our novel deep learning based solution can automatically align fruits and hence gives the ability to differentiate between length and width of round and blocky vegetables like tomatoes, onions and blocky peppers. After this fruit alignment/orientation correction different traits/phenotypes of vegetables can be measured easily. We achieved the orientation correction by novel framing of orientation correction problem as key point detection problem, we train a deep learning model to detect two key points per fruit and by measuring the angle of the line made by joining the points with horizontal axis and virtually rotating fruit by negative of that angle

<figure markdown>
  ![Image Calibration](img/orientation.png){ align=left }
  <figcaption>Orientation Correction</figcaption>
</figure>