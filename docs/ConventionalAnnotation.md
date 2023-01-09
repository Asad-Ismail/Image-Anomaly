1. Identify all the object instances in the image and draw a tight polygon around each one to create a "mask". The mask should cover all pixels that belong to the object instance and exclude all pixels that belong to the background or other object instances
2. Use a different label for each object class. For example, if the image contains both fruits and color board, use one label for fruits and a different label for color board.
3. Use the annotation tool or software to create a pixel-level mask for each object instance within the bounding box. The mask should cover all pixels that belong to the object instance and exclude all pixels that belong to the background or other object instances.
4. If possible, try to annotate a diverse set of images that includes a wide range of object classes, sizes, and orientations. This will help the model generalize to a wider range of real-world situations
5. We use Makesense tool for annotation

<figure markdown>
  ![Pepper Annotation](img/makesense.gif){ align=left }
  <figcaption>MakeSense Annotation</figcaption>
</figure>