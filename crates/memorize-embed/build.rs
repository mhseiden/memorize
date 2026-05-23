fn main() {
    // The CoreML execution provider in ORT needs Apple frameworks at link
    // time. `ort-sys` links `Foundation` itself; the rest are on us.
    // Also: ORT's prebuilt CoreML binary uses `__isPlatformVersionAtLeast`,
    // which lives in libclang_rt.osx — point the linker at it.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        for fw in &["CoreML", "Foundation", "CoreFoundation"] {
            println!("cargo:rustc-link-lib=framework={fw}");
        }
        if let Ok(out) = std::process::Command::new("xcrun")
            .args(["--find", "clang"])
            .output()
        {
            if let Ok(clang_path) = String::from_utf8(out.stdout) {
                let clang_path = clang_path.trim();
                if let Some(bin_dir) = std::path::Path::new(clang_path).parent() {
                    if let Some(usr_dir) = bin_dir.parent() {
                        if let Ok(entries) = std::fs::read_dir(usr_dir.join("lib/clang")) {
                            for entry in entries.flatten() {
                                let darwin = entry.path().join("lib/darwin");
                                if darwin.exists() {
                                    println!(
                                        "cargo:rustc-link-search=native={}",
                                        darwin.display()
                                    );
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }
}
