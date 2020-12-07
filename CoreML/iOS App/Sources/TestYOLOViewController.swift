//
//  TestYOLOViewController.swift
//  CS498
//
//  Created by Minghao Jiang on 11/27/20.
//  Copyright © 2020 MachineThink. All rights reserved.
//

import UIKit
import CoreML
import Vision

/**
  View controller for the "Camera" screen.

  This can evaluate both the neural network model and the k-NN model, because
  they use the same input and output names.

  On the Simulator this uses the photo library instead of the camera.
 */
class TestYOLOViewController: UIViewController {
  @IBOutlet var cameraButton: UIBarButtonItem!
  @IBOutlet var imageView: UIImageView!
  @IBOutlet var textView: UITextView!

  var model: MLModel!
  var predictor: YOLO!
    
    var boundingBoxes = [BoundingBox]()
    var colors: [UIColor] = []
    var secondStagePredictor: Predictor!

  override func viewDidLoad() {
    super.viewDidLoad()
    textView.text = "Use the camera to take a photo."
    predictor = YOLO()
    secondStagePredictor = Predictor(model: model)
    setUpBoundingBoxes()
  }
    
    let yoloLabels = [
      "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
      "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
      "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

  @IBAction func takePicture() {
    let picker = UIImagePickerController()
    picker.delegate = self
    picker.sourceType = UIImagePickerController.isSourceTypeAvailable(.camera) ? .camera : .photoLibrary
    picker.allowsEditing = true
    present(picker, animated: true)
  }
    
    func setUpBoundingBoxes() {
      for _ in 0..<YOLO.maxBoundingBoxes {
        boundingBoxes.append(BoundingBox())
      }

      // Make colors for the bounding boxes. There is one color for each class,
      // 20 classes in total.
      for r: CGFloat in [0.2, 0.4, 0.6, 0.8, 1.0] {
        for g: CGFloat in [0.3, 0.7] {
          for b: CGFloat in [0.4, 0.8] {
            let color = UIColor(red: r, green: g, blue: b, alpha: 1)
            colors.append(color)
          }
        }
      }
        
        for box in self.boundingBoxes {
          box.addToLayer(self.imageView.layer)
        }
    }

  func predict(image: UIImage) {
    // In the "Evaluate" screen we load the image from the dataset but here
    // we use a UIImage / CGImage object.
    if let pixelBuffer = image.pixelBuffer(width: YOLO.inputWidth, height: YOLO.inputHeight) {
        let predictions = predictor.predict(image: pixelBuffer)!
//        textView.text = makeDisplayString(predictions)
        var displayString = ""
        for i in 0..<boundingBoxes.count {
          if i < predictions.count {
            let prediction = predictions[i]

            // The predicted bounding box is in the coordinate space of the input
            // image, which is a square image of 416x416 pixels. We want to show it
            // on the video preview, which is as wide as the screen and has a 16:9
            // aspect ratio. The video preview also may be letterboxed at the top
            // and bottom.
            let height = imageView.bounds.height
            let width = imageView.bounds.width
            let scaleX = width / CGFloat(YOLO.inputWidth)
            let scaleY = height / CGFloat(YOLO.inputHeight)
            let top = (imageView.bounds.height - height) / 2

            // Translate and scale the rectangle to our own coordinate system.
            var rect = prediction.rect
            rect.origin.x *= scaleX
            rect.origin.y *= scaleY
            rect.origin.y += top
            rect.size.width *= scaleX
            rect.size.height *= scaleY
            
            
            let constraint = imageConstraint(model: model)

            let imageOptions: [MLFeatureValue.ImageOption: Any] = [
              .cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue
            ]
            
            let tSize = CGSize(width: YOLO.inputWidth, height: YOLO.inputHeight)
            let cropedImage = resizeImage(image: image, targetSize: tSize).cgImage!.cropping(to: prediction.rect)!
            let featureValue = (try? MLFeatureValue(cgImage: cropedImage,
                                                    orientation: .up,
                                                    constraint: constraint,
                                                    options: imageOptions))!
            let secondStagePrediction = secondStagePredictor.predict(image: featureValue)
            displayString += makeSecondStageDisplayString(secondStagePrediction!)

            // Show the bounding box.
//            let label = String(format: "%@ %.1f", yoloLabels[prediction.classIndex], prediction.score * 100)
            let label = String(format: "%@ %.1f", secondStagePrediction!.label, secondStagePrediction!.confidence * 100)
            let color = colors[prediction.classIndex]
            if yoloLabels[prediction.classIndex] == "person"{
                boundingBoxes[i].show(frame: rect, label: label, color: color)
            }
            else {
                boundingBoxes[i].hide()
            }
          } else {
            boundingBoxes[i].hide()
          }
        }
        textView.text = displayString
    }
  }
    
    private func makeSecondStageDisplayString(_ prediction: Prediction) -> String {
      var s = "Prediction: \(prediction.label)\n"
      s += String(format: "Probability: %.2f%%\n\n", prediction.confidence * 100)
      s += "Results for all classes:\n"
      s += prediction.sortedProbabilities
                     .map { String(format: "%@ %f", labels.userLabel(for: $0.0), $0.1) }
                     .joined(separator: "\n")
      return s
    }
  

private func makeDisplayString(_ prediction: [YOLO.Prediction]) -> String {
    var str = ""
    for p in prediction {
        var s = "Prediction: \(yoloLabels[p.classIndex])\n"
        s += String(format: "Probability: %.2f%%\n\n", p.score * 100)
        s += "\n"
        str += s
    }
    return str
    }
}

extension TestYOLOViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
  func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
    picker.dismiss(animated: true)

    let image = info[UIImagePickerController.InfoKey.editedImage] as! UIImage
    imageView.image = image

    predict(image: image)
  }
}


extension UIImage {
  public func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
    var maybePixelBuffer: CVPixelBuffer?
    let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                 kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue]
    let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                     Int(width),
                                     Int(height),
                                     kCVPixelFormatType_32ARGB,
                                     attrs as CFDictionary,
                                     &maybePixelBuffer)

    guard status == kCVReturnSuccess, let pixelBuffer = maybePixelBuffer else {
      return nil
    }

    CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
    let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer)

    guard let context = CGContext(data: pixelData,
                                  width: Int(width),
                                  height: Int(height),
                                  bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
    else {
      return nil
    }

    context.translateBy(x: 0, y: CGFloat(height))
    context.scaleBy(x: 1, y: -1)

    UIGraphicsPushContext(context)
    self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
    UIGraphicsPopContext()
    CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

    return pixelBuffer
  }
}

func resizeImage(image: UIImage, targetSize: CGSize) -> UIImage {
   let size = image.size

   let widthRatio  = targetSize.width  / size.width
   let heightRatio = targetSize.height / size.height

   // Figure out what our orientation is, and use that to form the rectangle
   var newSize: CGSize
   if(widthRatio > heightRatio) {
       newSize = CGSize(width: size.width * heightRatio, height: size.height * heightRatio)
   } else {
       newSize = CGSize(width: size.width * widthRatio,  height: size.height * widthRatio)
   }

   // This is the rect that we've calculated out and this is what is actually used below
   let rect = CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height)

   // Actually do the resizing to the rect using the ImageContext stuff
   UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
   image.draw(in: rect)
   let newImage = UIGraphicsGetImageFromCurrentImageContext()
   UIGraphicsEndImageContext()

   return newImage!
}
