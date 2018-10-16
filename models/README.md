You will need version 4.5.0 of KeYmaera X with mathematica configured.

You can get specific versions of KeYmaera X here:

    https://github.com/LS-Lab/KeYmaeraX-release/releases

KeYmaera X requires a decision procedure for real arithmetic to finalize proofs. 
It is tested best with Mathematica. 
After starting KeYmaera X you can configure arithmetic tools in the Help->Tool Configuration menu
after creating an acconut.

Once you have configured Mathematica, shut down the web ui by pressing the power button in the web ui.
Then you can verify the theorems in this folder by running the command:

`java -jar path/to/keymaerax.jar -prove acc_with_disturbance.kyx -tactic master.kyt -timeout 100`

`java -jar path/to/keymaerax.jar -prove ped.kyx -tactic master.kyt -timeout 100`

The ped proof might take a while. There's an optimized proof in `ped_proof.kyt`
