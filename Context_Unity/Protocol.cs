using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Gimbl;
using UnityEditor;

public class Protocol : MonoBehaviour
{
    // Assets.
    public GameObject CuePrefab;        // Cue Prefab used to create cues..
    public ActorObject actor;           // Task actor.
    // Services.
    public MQTTHandler mqtt;            // Handles MQTT messaging.
    public LogService log;              // Handles logging.
    // Task parameters.
    public ProtocolSettings settings = new ProtocolSettings();
    // Private.
    private float cueSize = 2f;         // Size of cue in VR units (decimeters).
    private Cue[] cues;                 // Stores array of all cues.

    public void InitializeTask(ProtocolSettings settings) 
    {
        // Change settings.
        this.settings = settings; 
        InitializeCues();
    }

    // Send message of Reward MQTT topic.
    public void Reward() { mqtt.SendReward(); log.LogReward(); } 

    // Called upon Lick MQTT message.
    public void OnLick() {
        // Log lick.
        log.LogLick();
        // Call OnLick function in every Cue Object.
        if (cues != null) 
        { 
            foreach (Cue cue in cues) { cue.OnLick(); }
        }
        
    }

    // Setup all cues along path according to task parameters.
    public void InitializeCues()
    {
        // Check if previous cues need to be removed.
        RemoveCues();
        // Calculate number of cues pased of cue distance parameter.
        int numCues = Mathf.FloorToInt(
            FindObjectOfType<PathCreation.PathCreator>().path.length / 
            (cueSize + settings.cueDistance));
        cues = new Cue[numCues];
        // Instantiate cues from prefab and set center position.
        float cuePos = 0f;
        for (int i = 0;i< numCues;i++)
        {
            cuePos += cueSize + settings.cueDistance;
            cues[i] = CreateCue(cuePos, i );
        }
    }

    public void RemoveCues()
    {
        foreach (Cue cue in FindObjectsOfType<Cue>())
        {
            Destroy(cue.gameObject);
        }
        cues = null;
    }

    /// <summary>
    /// Function that create a new cue.
    /// </summary>
    /// <param name="pos">center Z position of cue.</param>
    /// <param name="id">id of cue.</param>
    /// <returns></returns>
    private Cue CreateCue(float pos, int id)
    {
        // Create
        Cue cue = Instantiate(CuePrefab, new Vector3(0, 0, pos), Quaternion.identity).GetComponent<Cue>();
        cue.protocol = this;
        cue.log = log;
        cue.position = pos;
        cue.id = id;
        cue.id2 = 0;
        // Set reward state.
        cue.NewState();
        return cue;
    }
}

// Custom Inspector Window.
[CustomEditor(typeof(Protocol))]
public class SomeScriptEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector(); // Draw default inspector elements.
        // Add Redraw button.
        if (GUILayout.Button("Redraw all Cues")) { ((Protocol)target).InitializeCues(); }
    }
}