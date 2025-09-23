using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Gimbl;

public class Cue : MonoBehaviour
{
    public Protocol protocol;    // Reference to main protocol object.
    public Material grayMat;            // Material used for gray-out. 
    public int id;                      // Cue Identifier.
    public int id2 = 0;                 // flipped Identifier.
    public float position;              // Center position of Cue (result position = position + random offset)
    
    private bool passedThrough = false; // Flag if animal passed through cue.
    private Material cueMaterial;
    public bool isRewarding;            // Flag if cue is in rewarding state.
    public bool hasGivenReward = false; // Flagg if cue has dispensed reward.
    public bool isGray = true;          // Flag if cue 
    // Lick tracking
    public int numLicksInReward = 0;    // Counter for number of licks in reward zone.
    public int numLicksInPre = 0;       // Counter for number of licks in pre-reward zone.
    // Zone tracking variables.
    private bool inReward = false;      // Flag if animal is currently in reward zone.
    private bool inPre = false;         // Flag if animal is currently in pre-reward zone.

    // Services.
    public LogService log;

    // Set Reward state based on random reward probability.
    public void SetRewardState()
    {
        // check settings if reward is random or linked to cue texture.
        if (!protocol.settings.hasRandomRewards)
        {
            isRewarding = Random.value <= protocol.settings.rewardCueProb;
            cueMaterial = isRewarding ? protocol.settings.cueSet.rewardingMat : protocol.settings.cueSet.nonRewardingMat;
            id2 = 3;
        }
        // Decouple texture and reward startus if hasRandomRewards.
        else
        {
            isRewarding = Random.value <= protocol.settings.randomRewardChance;
            if (Random.value <= protocol.settings.rewardCueProb)
            {
                cueMaterial = protocol.settings.cueSet.rewardingMat;
                id2 = 1;
            }
            else
            {
                cueMaterial = protocol.settings.cueSet.nonRewardingMat;
                id2 = 2;
            }
            //cueMaterial = Random.value <= protocol.settings.rewardCueProb ? protocol.settings.cueSet.rewardingMat : protocol.settings.cueSet.nonRewardingMat;
        }
            isGray = protocol.settings.grayOutCues; // # set if cue starts out gray.
            UpdateTexture();
        }

    // Update cue texture based on current paramters.
    public void UpdateTexture()
    {
        if (isGray) { SetMaterial(grayMat); }
        else { SetMaterial(cueMaterial); }
    }

    // Set position cue based on center position and random offset.
    public void SetDistanceOffset()
    {
        float offset = Random.Range(-protocol.settings.maxDistOffset, protocol.settings.maxDistOffset);
        transform.position = new Vector3(0, 0, position + offset);
    }

    // Test if reward should be given based on current state.
    private void TestForReward(bool licked)
    {
        if (inReward && isRewarding && !hasGivenReward && protocol.actor.isActive) // Test if cue can still give reward.
        {
            if (licked && !protocol.settings.isGuided) { Reward(); }        // Non-guided and animal licked.
            if (protocol.settings.isGuided) { Reward(); }        // "must lick" negative and no lick.
        }
    }

    // Called when MQTT lick message is received.
    public void OnLick() {
        // Test for reward with lick.
        TestForReward(true); 
        // Track Lick counter based on position.
        if (inPre) { numLicksInPre++; }
        if (inReward) { numLicksInReward++; }
    }

    // Assign new state and position to Cue.
    public void NewState()
    {
        passedThrough = false;
        hasGivenReward = false;
        numLicksInReward = 0;
        numLicksInPre = 0;
        SetRewardState();
        SetDistanceOffset();
        log.LogState(this);
    }

    // Dispense reward.
    public void Reward()
    {
        hasGivenReward = true;
        protocol.Reward();
    }

    // Set Material/texture of cue.
    public void SetMaterial(Material mat)
    {
        foreach (Renderer render in this.gameObject.GetComponentsInChildren<Renderer>()){ render.material = mat;}
    }

    // Called upon entering Reward zone.
    public void OnRewardEnter()
    {
        passedThrough = true;
        inReward = true;
        TestForReward(false); // test for reward without lick.
    }

    // Called upon exiting Reward zone.
    public void OnRewardExit() { inReward = false; }

    // Called upon enter reset zone.
    public void OnResetEnter()
    {
        if (passedThrough)
        {
            log.LogResult(this);    // Log results of traversal.
            NewState();     // Set new state/position of cue.
        }
    }

    // Called upon entering pre-reward zone.
    public void OnPreEnter() {inPre = true; }
    // Called upon exiting pre-reward zone.
    public void OnPreExit()  {inPre = false;}
    // Called upon entering visible zone.
    public void OnVisibleEnter() { if (protocol.settings.grayOutCues) { isGray = false; UpdateTexture(); } }
    // Called upon exiting visible zone.
    public void OnVisibleExit() { if (protocol.settings.grayOutCues) { isGray = true; UpdateTexture(); } }

}
