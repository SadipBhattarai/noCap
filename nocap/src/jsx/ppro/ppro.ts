export const getSelectedClipPath = (): any => {
  try {
    if (!app || !app.project || !app.project.activeSequence) {
      return { error: "No active sequence found." };
    }

    var seq = app.project.activeSequence;
    var selection = seq.getSelection();

    if (!selection || selection.length === 0) {
      return { error: "Select a clip on the timeline." };
    }

    var clip = selection[0];
    if (!clip || !clip.projectItem) {
      return { error: "Selected clip has no linked project media." };
    }

    var path = "";
    if (clip.projectItem.getMediaPath) {
      path = clip.projectItem.getMediaPath();
    }

    if (!path) {
      return { error: "Selected clip has no accessible media path." };
    }

    var startSeconds = 0;
    if (clip.start && clip.start.seconds) {
      startSeconds = clip.start.seconds;
    }

    return { path: path, startSeconds: startSeconds };
  } catch (e: any) {
    return { error: "ExtendScript Error: " + e.toString() };
  }
};

export const importSRT = (filePath: string, startSeconds: number): string => {
  try {
    if (!app || !app.project || !app.project.activeSequence)
      return "ERROR: No active sequence.";

    var srtFile = new File(filePath);
    if (!srtFile.exists) return "ERROR: SRT file not found on disk.";

    // Import the file directly to the root bin
    app.project.importFiles(
      [srtFile.fsName],
      true,
      app.project.rootItem,
      false,
    );

    // Grab the most recently imported SRT file from the root bin directly
    var root = app.project.rootItem;
    var srtItem = null;

    if (root && root.children) {
      for (var i = 0; i < root.children.numItems; i++) {
        var child = root.children[i];
        if (child && child.name && child.name.indexOf(".srt") !== -1) {
          srtItem = child; // Overwrites until it finds the latest one
        }
      }
    }

    if (!srtItem)
      return "ERROR: Could not locate imported SRT in the project bin.";

    // 0 is the default enum for Subtitle format in Premiere
    app.project.activeSequence.createCaptionTrack(
      srtItem,
      startSeconds || 0,
      0,
    );

    return "SUCCESS";
  } catch (e: any) {
    return "ERROR: " + e.toString();
  }
};
