import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, UpdateImagesContext, PromptContext, NegativePromptContext, DeletionContext,
StepsContext, CFGContext, ModelNameContext, SizeContext, SamplerContext, DenoiseContext, InterrogateTextContext, UpdateSavedContext, ClipSkipContext, PreviewImageContext, TabContext,
SeedContext, VAENameContext, ImageBrightnessContext, ImageContrastContext, GeneratorContext} from "../Context"
import functions from "../structures/Functions"
import deleteIcon from "../assets/icons/delete.png"
import deleteIconHover from "../assets/icons/delete-hover.png"
import saveIcon from "../assets/icons/save.png"
import saveIconHover from "../assets/icons/save-hover.png"
import sendIcon from "../assets/icons/send.png"
import sendIconHover from "../assets/icons/send-hover.png"
import imagesMeta from "images-meta"
import path from "path"
import "./styles/image.less"
import axios from "axios"

interface ImageHistoryImageProps {
    img: string
    small?: boolean
}

let timer = null as any
let clicking = false

const Image: React.FunctionComponent<ImageHistoryImageProps> = (props) => {
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {imageBrightness, setImageBrightness} = useContext(ImageBrightnessContext)
    const {imageContrast, setImageContrast} = useContext(ImageContrastContext)
    const [deleteHover, setDeleteHover] = useState(false)
    const [saveHover, setSaveHover] = useState(false)
    const [sendHover, setSendHover] = useState(false)
    const {updateImages, setUpdateImages} = useContext(UpdateImagesContext)
    const {steps, setSteps} = useContext(StepsContext)
    const {cfg, setCFG} = useContext(CFGContext)
    const {size, setSize} = useContext(SizeContext)
    const {denoise, setDenoise} = useContext(DenoiseContext)
    const {sampler, setSampler} = useContext(SamplerContext)
    const {interrogateText, setInterrogateText} = useContext(InterrogateTextContext)
    const {prompt, setPrompt} = useContext(PromptContext)
    const {negativePrompt, setNegativePrompt} = useContext(NegativePromptContext)
    const {clipSkip, setClipSkip} = useContext(ClipSkipContext)
    const {modelName, setModelName} = useContext(ModelNameContext)
    const {updateSaved, setUpdateSaved} = useContext(UpdateSavedContext)
    const {previewImage, setPreviewImage} = useContext(PreviewImageContext)
    const {seed, setSeed} = useContext(SeedContext)
    const {vaeName, setVAEName} = useContext(VAENameContext)
    const {deletion, setDeletion} = useContext(DeletionContext)
    const {generator, setGenerator} = useContext(GeneratorContext)
    const {tab, setTab} = useContext(TabContext)
    const [isSaved, setIsSaved] = useState(false)
    const [hover, setHover] = useState(false)

    const getSaveKey = () => {
        if (generator === "novel ai") return "saved-novel-ai"
        if (generator === "holara ai") return "saved-holara-ai"
        return "saved"
    }

    const updateSavedImages = () => {
        let saved = localStorage.getItem(getSaveKey()) || "[]" as any
        saved = JSON.parse(saved)
        if (saved.includes(props.img)) {
            setIsSaved(true)
        } else {
            setIsSaved(false)
        }
    }

    useEffect(() => {
        updateSavedImages()
    }, [updateSaved, generator])

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    const sendImage = async (event: any) => {
        event.stopPropagation()
        if (!hover) return
        let inMime = "image/jpeg"
        if (path.extname(props.img) === ".png") inMime = "image/png"
        if (path.extname(props.img) === ".webp") inMime = "image/webp"
        const arrayBuffer = await fetch((props.img)).then((r) => r.arrayBuffer())
        if (inMime === "image/png") {
            let arr = [] as any
            const meta = imagesMeta.readMeta(Buffer.from(arrayBuffer), inMime)
            for (let i = 0; i < meta.length; i++) {
                if (meta[i].name?.toLowerCase() === "prompt") {
                    arr.push(`Prompt: ${meta[i].value}`)
                } else if (meta[i].name?.toLowerCase() === "negative prompt") {
                    arr.push(`Negative Prompt: ${meta[i].value}`)
                } else if (meta[i].name?.toLowerCase() === "size") {
                    arr.push(`Size: ${meta[i].value}`)
                } else if (meta[i].name?.toLowerCase() === "model") {
                    arr.push(`Model: ${meta[i].value}`)
                } else if (meta[i].name?.toLowerCase() === "vae") {
                    arr.push(`VAE: ${meta[i].value}`)
                } else if (meta[i].name?.toLowerCase() === "steps") {
                    arr.push(`Steps: ${meta[i].value}`)
                } else if (meta[i].name?.toLowerCase() === "cfg") {
                    arr.push(`CFG: ${meta[i].value}`)
                } else if (meta[i].name?.toLowerCase() === "denoise") {
                    arr.push(`Denoise: ${meta[i].value}`)
                } else if (meta[i].name?.toLowerCase() === "sampler") {
                    arr.push(`Sampler: ${meta[i].value}`)
                } else if (meta[i].name?.toLowerCase() === "clip skip") {
                    arr.push(`Clip Skip: ${meta[i].value}`)
                } else if (meta[i].name?.toLowerCase() === "seed") {
                    arr.push(`Seed: ${meta[i].value}`)
                }
            }
            setInterrogateText(arr.join("\n"))
        } else if (inMime === "image/jpeg") {
            let str = ""
            const meta = imagesMeta.readMeta(Buffer.from(arrayBuffer), inMime)
            for (let i = 0; i < meta.length; i++) {
                if (meta[i].name?.toLowerCase() === "usercomment") {
                    str +=  meta[i].value.replaceAll("UNICODE", "").replaceAll(/\u0000/g, "")
                }
            }
            const {prompt, negativePrompt, size, model, vae, denoise, steps, cfg, sampler, clipSkip, seed} = functions.extractMetaValues(str)
            let arr = [] as any
            if (prompt) {
                arr.push(`Prompt: ${prompt}`)
            } if (negativePrompt) {
                arr.push(`Negative Prompt: ${negativePrompt}`)
            } if (size) {
                arr.push(`Size: ${size}`)
            } if (model) {
                arr.push(`Model: ${model}`)
            } if (vae) {
                arr.push(`VAE: ${vae}`)
            } if (steps) {
                arr.push(`Steps: ${steps}`)
            } if (cfg) {
                arr.push(`CFG: ${cfg}`)
            } if (denoise) {
                arr.push(`Denoise: ${denoise}`)
            } if (sampler) {
                arr.push(`Sampler: ${sampler}`)
            } if (clipSkip) {
                arr.push(`Clip Skip: ${clipSkip}`)
            } if (seed) {
                arr.push(`Seed: ${seed}`)
            }
            setInterrogateText(arr.join("\n"))
        } else {
            const form = new FormData()
            const blob = new Blob([new Uint8Array(arrayBuffer)])
            const file = new File([blob], props.img)
            form.append("image", file)
            const exif = await axios.post("get-exif", form).then((r) => r.data)
            const str = exif.replaceAll("UNICODE", "").replaceAll(/\u0000/g, "")
            const {prompt, negativePrompt, size, model, vae, denoise, steps, cfg, sampler, clipSkip, seed} = functions.extractMetaValues(str)
            let arr = [] as any
            if (prompt) {
                arr.push(`Prompt: ${prompt}`)
            } if (negativePrompt) {
                arr.push(`Negative Prompt: ${negativePrompt}`)
            } if (size) {
                arr.push(`Size: ${size}`)
            } if (model) {
                arr.push(`Model: ${model}`)
            } if (vae) {
                arr.push(`VAE: ${vae}`)
            } if (steps) {
                arr.push(`Steps: ${steps}`)
            } if (cfg) {
                arr.push(`CFG: ${cfg}`)
            } if (denoise) {
                arr.push(`Denoise: ${denoise}`)
            } if (sampler) {
                arr.push(`Sampler: ${sampler}`)
            } if (clipSkip) {
                arr.push(`Clip Skip: ${clipSkip}`)
            } if (seed) {
                arr.push(`Seed: ${seed}`)
            }
            setInterrogateText(arr.join("\n"))
        }
        setTab("generate")
    }

    const saveImage = (event: any) => {
        event.stopPropagation()
        if (!hover) return
        let saved = localStorage.getItem(getSaveKey()) || "[]" as any
        saved = JSON.parse(saved)
        const exists = saved.find((i: string) => i === props.img)
        if (exists) {
            saved = functions.removeItem(saved, props.img)
            localStorage.setItem(getSaveKey(), JSON.stringify(saved))
            setIsSaved(false)
        } else {
            saved.push(props.img)
            localStorage.setItem(getSaveKey(), JSON.stringify(saved))
            setIsSaved(true)
        }
        setUpdateSaved(true)
        axios.post("/save-images", {saved, generator})
    }

    const deleteImage = async (event: any) => {
        event.stopPropagation()
        if (!hover) return
        await axios.post("/delete-file", {path: props.img, deletion})
        setUpdateImages(true)
    }

    const showInFolder = () => {
        axios.post("/show-in-folder", {path: props.img})
    }

    const preview = () => {
        setPreviewImage(props.img)
    }

    const handleClick = (event: any) => {
        if (previewImage) return clearTimeout(timer)
        if (clicking) {
            clicking = false
            clearTimeout(timer)
            return showInFolder()
        }
        clicking = true
        timer = setTimeout(() => {
            clicking = false
            clearTimeout(timer)
            preview()
        }, 200)
    }

    const getSaveIcon = () => {
        if (isSaved) {
            return saveIconHover
        } else {
            return saveHover ? saveIconHover : saveIcon
        }
    }

    const getMaxHeight = () => {
        if (generator === "novel ai") return "420px"
        if (generator === "holara ai") return "420px"
        return "465px"
    }

    if (props.small) {
        return (
            <div className="image-img-container-small" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)} onClick={handleClick}>
                <div className={`image-img-button-container-small ${hover ? "image-buttons-show" : ""}`}>
                    <img className="image-img-button-small" onMouseEnter={() => setSendHover(true)} onClick={sendImage} draggable={false}
                    onMouseLeave={() => setSendHover(false)} src={sendHover ? sendIconHover : sendIcon} style={{filter: getFilter(), cursor: hover ? "pointer" : "default"}}/>
                    <img className="image-img-button-small" onMouseEnter={() => setSaveHover(true)} onClick={saveImage} draggable={false}
                    onMouseLeave={() => setSaveHover(false)} src={getSaveIcon()} style={{filter: getFilter(), cursor: hover ? "pointer" : "default"}}/>
                    <img className="image-img-button-small" onMouseEnter={() => setDeleteHover(true)} onClick={deleteImage} draggable={false}
                    onMouseLeave={() => setDeleteHover(false)} src={deleteHover ? deleteIconHover : deleteIcon} style={{filter: getFilter(), cursor: hover ? "pointer" : "default"}}/>
                </div>
                <img className="image-img-small" src={props.img} draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%)`}}/>
            </div>
        )
    }

    return (
        <div className="image-img-container" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)} onClick={handleClick}>
            <div className={`image-img-button-container ${hover ? "image-buttons-show" : ""}`}>
                <img className="image-img-button" onMouseEnter={() => setSendHover(true)} onClick={sendImage} draggable={false}
                onMouseLeave={() => setSendHover(false)} src={sendHover ? sendIconHover : sendIcon} style={{filter: getFilter(), cursor: hover ? "pointer" : "default"}}/>
                <img className="image-img-button" onMouseEnter={() => setSaveHover(true)} onClick={saveImage} draggable={false}
                onMouseLeave={() => setSaveHover(false)} src={getSaveIcon()} style={{filter: getFilter(), cursor: hover ? "pointer" : "default"}}/>
                <img className="image-img-button" onMouseEnter={() => setDeleteHover(true)} onClick={deleteImage} draggable={false}
                onMouseLeave={() => setDeleteHover(false)} src={deleteHover ? deleteIconHover : deleteIcon} style={{filter: getFilter(), cursor: hover ? "pointer" : "default"}}/>
            </div>
            <img className="image-img" src={props.img} draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%)`, maxHeight: getMaxHeight()}}/>
        </div>
    )
}

export default Image