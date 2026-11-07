// ExtendScript to find the file path of the selected clip on the timeline
function getSelectedClipPath() {
  var sequence = app.project.activeSequence;
  if (!sequence) {
    alert("Please open a sequence.");
    return null;
  }

  // Check video tracks first
  for (var i = 0; i < sequence.videoTracks.numTracks; i++) {
    var track = sequence.videoTracks[i];
    for (var j = 0; j < track.clips.numItems; j++) {
      var clip = track.clips[j];
      if (clip.isSelected()) {
        var path = clip.projectItem.getMediaPath();
        alert("Selected File Path:\n" + path);
        return path;
      }
    }
  }

  // Check audio tracks if no video clip is selected
  for (var k = 0; k < sequence.audioTracks.numTracks; k++) {
    var aTrack = sequence.audioTracks[k];
    for (var l = 0; l < aTrack.clips.numItems; l++) {
      var aClip = aTrack.clips[l];
      if (aClip.isSelected()) {
        var aPath = aClip.projectItem.getMediaPath();
        alert("Selected File Path:\n" + aPath);
        return aPath;
      }
    }
  }

  alert("Please select a video or audio clip on the timeline first.");
  return null;
}

// Execute the function
getSelectedClipPath();
