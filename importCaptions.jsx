// ExtendScript to import a generated SRT file into Premiere Pro
function importSRT(filePath) {
  var project = app.project;
  if (!project) {
    alert("Please open a project first.");
    return;
  }

  // Import the SRT file into the root of the project
  var importSuccess = project.importFiles(
    [filePath],
    1, // suppress warnings
    project.rootItem,
    0, // ignore folder structure
  );

  if (importSuccess) {
    alert(
      "Captions imported successfully! Drag the SRT from your project bin to the timeline.",
    );
  } else {
    alert("Failed to import captions.");
  }
}

// In a real CEP panel, your frontend passes the downloaded SRT path here:
importSRT("/Users/Shared/generated_captions.srt");
