"""Official evaluation of the MICCAI WMH Segmentation Challenge, vendored verbatim (J-012).

Source: https://github.com/hjkuijf/wmhchallenge, file ``evaluation.py``, commit
2e628cbe5a2da37d9be24d455fdaee7438adf9c5 (fetched 2026-10-04). The functions are kept as published
(only the command-line wrapper ``do()`` and its fixed folder names are dropped); their
code is unchanged, so our numbers are computed exactly as the challenge leaderboard.

Original license:

    MIT License

    Copyright (c) 2020 Hugo J. Kuijf

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

# ruff: noqa  (third-party code kept verbatim)

import difflib
import os

import numpy as np
import scipy.spatial
import SimpleITK as sitk


def getImages(testFilename, resultFilename):
    """Return the test and result images, thresholded and non-WMH masked."""
    testImage = sitk.ReadImage(testFilename)
    resultImage = sitk.ReadImage(resultFilename)

    # Check for equality
    assert testImage.GetSize() == resultImage.GetSize()

    # Get meta data from the test-image, needed for some sitk methods that check this
    resultImage.CopyInformation(testImage)

    # Remove non-WMH from the test and result images, since we don't evaluate on that
    maskedTestImage = sitk.BinaryThreshold(testImage, 0.5, 1.5, 1, 0)  # WMH == 1
    nonWMHImage = sitk.BinaryThreshold(testImage, 1.5, 2.5, 0, 1)  # non-WMH == 2
    maskedResultImage = sitk.Mask(resultImage, nonWMHImage)

    # Convert to binary mask
    if "integer" in maskedResultImage.GetPixelIDTypeAsString():
        bResultImage = sitk.BinaryThreshold(maskedResultImage, 1, 1000, 1, 0)
    else:
        bResultImage = sitk.BinaryThreshold(maskedResultImage, 0.5, 1000, 1, 0)

    return maskedTestImage, bResultImage


def getResultFilename(participantDir):
    """Find the filename of the result image.

    This should be result.nii.gz or result.nii. If these files are not present,
    it tries to find the closest filename."""
    files = os.listdir(participantDir)

    if not files:
        raise Exception("No results in " + participantDir)

    resultFilename = None
    if "result.nii.gz" in files:
        resultFilename = os.path.join(participantDir, "result.nii.gz")
    elif "result.nii" in files:
        resultFilename = os.path.join(participantDir, "result.nii")
    else:
        # Find the filename that is closest to 'result.nii.gz'
        maxRatio = -1
        for f in files:
            currentRatio = difflib.SequenceMatcher(a=f, b="result.nii.gz").ratio()

            if currentRatio > maxRatio:
                resultFilename = os.path.join(participantDir, f)
                maxRatio = currentRatio

    return resultFilename


def getDSC(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testArray = sitk.GetArrayFromImage(testImage).flatten()
    resultArray = sitk.GetArrayFromImage(resultImage).flatten()

    # similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray)


def getHausdorff(testImage, resultImage):
    """Compute the Hausdorff distance."""

    # Hausdorff distance is only defined when something is detected
    resultStatistics = sitk.StatisticsImageFilter()
    resultStatistics.Execute(resultImage)
    if resultStatistics.GetSum() == 0:
        return float("nan")

    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
    eTestImage = sitk.BinaryErode(testImage, (1, 1, 0))
    eResultImage = sitk.BinaryErode(resultImage, (1, 1, 0))

    hTestImage = sitk.Subtract(testImage, eTestImage)
    hResultImage = sitk.Subtract(resultImage, eResultImage)

    hTestArray = sitk.GetArrayFromImage(hTestImage)
    hResultArray = sitk.GetArrayFromImage(hResultImage)

    # Convert voxel location to world coordinates. Use the coordinate system of the test image
    # np.nonzero   = elements of the boundary in numpy order (zyx)
    # np.flipud    = elements in xyz order
    # np.transpose = create tuples (x,y,z)
    # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
    testCoordinates = [
        testImage.TransformIndexToPhysicalPoint(x.tolist())
        for x in np.transpose(np.flipud(np.nonzero(hTestArray)))
    ]
    resultCoordinates = [
        testImage.TransformIndexToPhysicalPoint(x.tolist())
        for x in np.transpose(np.flipud(np.nonzero(hResultArray)))
    ]

    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]

    # Compute distances from test to result; and result to test
    dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
    dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)

    return max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))


def getLesionDetection(testImage, resultImage):
    """Lesion detection metrics, both recall and F1."""

    # Connected components will give the background label 0, so subtract 1 from all results
    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(True)

    # Connected components on the test image, to determine the number of true WMH.
    # And to get the overlap between detected voxels and true WMH
    ccTest = ccFilter.Execute(testImage)
    lResult = sitk.Multiply(ccTest, sitk.Cast(resultImage, sitk.sitkUInt32))

    ccTestArray = sitk.GetArrayFromImage(ccTest)
    lResultArray = sitk.GetArrayFromImage(lResult)

    # recall = (number of detected WMH) / (number of true WMH)
    nWMH = len(np.unique(ccTestArray)) - 1
    if nWMH == 0:
        recall = 1.0
    else:
        recall = float(len(np.unique(lResultArray)) - 1) / nWMH

    # Connected components of results, to determine number of detected lesions
    ccResult = ccFilter.Execute(resultImage)
    lTest = sitk.Multiply(ccResult, sitk.Cast(testImage, sitk.sitkUInt32))

    ccResultArray = sitk.GetArrayFromImage(ccResult)
    lTestArray = sitk.GetArrayFromImage(lTest)

    # precision = (number of detections that intersect with WMH) / (number of all detections)
    nDetections = len(np.unique(ccResultArray)) - 1
    if nDetections == 0:
        precision = 1.0
    else:
        precision = float(len(np.unique(lTestArray)) - 1) / nDetections

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return recall, f1


def getAVD(testImage, resultImage):
    """Volume statistics."""
    # Compute statistics of both images
    testStatistics = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()

    testStatistics.Execute(testImage)
    resultStatistics.Execute(resultImage)

    return (
        float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) / float(testStatistics.GetSum()) * 100
    )
