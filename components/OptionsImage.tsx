import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, DrawImageContext, MaskImageContext,
InterrogateTextContext, ImageInputContext, ExpandImageContext, MaskDataContext, ImageBrightnessContext, ImageContrastContext, ExpandDialogFlagContext, 
HorizontalExpandContext, VerticalExpandContext, ExpandMaskContext} from "../Context"
import functions from "../structures/Functions"
import imgPlaceHolder from "../assets/images/img-placeholder.png"
import fileType from "magic-bytes.js"
import xIcon from "../assets/icons/x-alt.png"
import drawIcon from "../assets/icons/draw.png"
import segmentateIcon from "../assets/icons/segmentate.png"
import expandIcon from "../assets/icons/expand.png"
import path from "path"
import axios from "axios"
import "./styles/optionsbar.less"

const OptionsImage: React.FunctionComponent = (props) => {
    const {imageInput, setImageInput} = useContext(ImageInputContext)
    const [hover, setHover] = useState(false)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {imageBrightness, setImageBrightness} = useContext(ImageBrightnessContext)
    const {imageContrast, setImageContrast} = useContext(ImageContrastContext)
    const {interrogateText, setInterrogateText} = useContext(InterrogateTextContext)
    const {drawImage, setDrawImage} = useContext(DrawImageContext)
    const {maskImage, setMaskImage} = useContext(MaskImageContext)
    const {maskData, setMaskData} = useContext(MaskDataContext)
    const {expandDialogFlag, setExpandDialogFlag} = useContext(ExpandDialogFlagContext)
    const {expandImage, setExpandImage} = useContext(ExpandImageContext)
    const {expandMask, setExpandMask} = useContext(ExpandMaskContext)
    const {horizontalExpand, setHorizontalExpand} = useContext(HorizontalExpandContext)
    const {verticalExpand, setVerticalExpand} = useContext(VerticalExpandContext)
    const [img, setImg] = useState(null) as any
    const [maskImg, setMaskImg] = useState(null) as any 
    const ref = useRef(null) as any

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedImage = localStorage.getItem("imageInput")
        if (savedImage) setImageInput(savedImage)
    }, [])

    useEffect(() => {
        localStorage.setItem("imageInput", String(imageInput))
    }, [imageInput])

    const loadImage = async (event: any) => {
        const file = event.target.files?.[0]
        if (!file) return
        const fileReader = new FileReader()
        await new Promise<void>((resolve) => {
            fileReader.onloadend = async (f: any) => {
                let bytes = new Uint8Array(f.target.result)
                const result = fileType(bytes)?.[0]
                const jpg = result?.mime === "image/jpeg" ||
                path.extname(file.name).toLowerCase() === ".jpg" ||
                path.extname(file.name).toLowerCase() === ".jpeg"
                const png = result?.mime === "image/png"
                const webp = result?.mime === "image/webp"
                if (jpg) result.typename = "jpg"
                if (jpg || png || webp) {
                    const url = functions.arrayBufferToBase64(bytes.buffer)
                    const link = `${url}#.${result.typename}`
                    removeImage()
                    const metadata = await functions.getImageMetaData(link)
                    if (metadata) setInterrogateText(metadata)
                    setTimeout(() => {
                        setImageInput(link)
                    }, 100)
                }
                resolve()
            }
            fileReader.readAsArrayBuffer(file)
        })
        if (event.target) event.target.value = ""
    }

    const removeImage = (event?: any) => {
        event?.preventDefault()
        event?.stopPropagation()
        setImageInput("")
        setMaskImage("")
        setMaskData("")
        setImg(null)
        setMaskImg(null)
        setHorizontalExpand("0")
        setVerticalExpand("0")
        setExpandImage("")
        setExpandMask("")
    }

    const startDrawing = (event: any) => {
        if (!imageInput) return
        event?.preventDefault()
        event?.stopPropagation()
        setDrawImage(imageInput)
    }

    const segmentate = async (event: any) => {
        if (!imageInput) return
        event?.preventDefault()
        event?.stopPropagation()
        const form = new FormData()
        const arrayBuffer = await fetch(imageInput).then((r) => r.arrayBuffer())
        const blob = new Blob([new Uint8Array(arrayBuffer)])
        const file = new File([blob], "image.png", {type: "image/png"})
        form.append("image", file)
        await axios.post("/segmentate", form)
    }

    const expandDialog = () => {
        if (!imageInput) return
        event?.preventDefault()
        event?.stopPropagation()
        setExpandDialogFlag(true)
    }

    const loadImages = async () => {
        if (!imageInput) return
        const image = document.createElement("img")
        await new Promise<void>((resolve) => {
            image.onload = () => resolve()
            image.src = imageInput
        })
        setImg(image)
        if (!maskImage) return
        const mask = document.createElement("img")
        await new Promise<void>(async (resolve) => {
            mask.onload = () => resolve()
            mask.src = maskImage
        })
        setMaskImg(mask)
    }
    useEffect(() => {
        loadImages()
    }, [imageInput, maskImage])

    const getNormalizedDimensions = () => {
        let greaterValue = img.width > img.height ? img.width : img.height
        const heightBigger = img.height > img.width
        const ratio = greaterValue / (heightBigger ? 800 : 1200)
        const width = Math.floor(img.width / ratio)
        const height = Math.floor(img.height / ratio)
        return {width, height}
    }

    const updateImage = () => {
        if (!ref.current || !img) return
        ref.current.width = getNormalizedDimensions().width
        ref.current.height = getNormalizedDimensions().height
        const ctx = ref.current.getContext("2d")!
        ctx.drawImage(img, 0, 0, ref.current.width, ref.current.height)

        if (maskImg) {
            const maskCanvas = document.createElement("canvas")
            maskCanvas.width = maskImg.width 
            maskCanvas.height = maskImg.height
            const maskCtx = maskCanvas.getContext("2d")!
            maskCtx.drawImage(maskImg, 0, 0, ref.current.width, ref.current.height)

            const imgData = ctx.getImageData(0, 0, ref.current.width, ref.current.height)
            const maskPixels = maskCtx.getImageData(0, 0, ref.current.width, ref.current.height).data

            const pixels = imgData.data
            for (let i = 0; i < pixels.length; i+=4) {
                if (maskPixels[i] < 10 && maskPixels[i+1] < 10 && maskPixels[i+2] < 10) {
                    // ignore
                } else {
                    const color = functions.rotateColor("#fc1594", siteHue, siteSaturation, siteLightness)
                    const {r, g, b} = functions.hexToRgb(color)
                    pixels[i] = r 
                    pixels[i+1] = g
                    pixels[i+2] = b
                }
            }
            ctx.putImageData(imgData, 0, 0)
        }

        const horizontal = Math.round(Number(horizontalExpand))
        const vertical = Math.round(Number(verticalExpand))
        if (!Number.isNaN(horizontal) && !Number.isNaN(vertical) && !(horizontal === 0 && vertical === 0)) {
            const imgData = ctx.getImageData(0, 0, ref.current.width, ref.current.height)
            ctx.clearRect(0, 0, ref.current.width, ref.current.height)
            ref.current.width = ref.current.width + Math.floor(horizontal * 2)
            ref.current.height = ref.current.height + Math.floor(vertical * 2)
            ctx.fillStyle = "white"
            ctx.fillRect(0, 0, ref.current.width, ref.current.height)
            ctx.putImageData(imgData, horizontal, vertical, 0, 0, ref.current.width, ref.current.height)
        }
    }

    useEffect(() => {
        updateImage()
    }, [img, maskImg, siteHue, siteSaturation, siteLightness, horizontalExpand, verticalExpand])

    return (<>
        <label htmlFor="img" className="options-bar-img-container" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
            <div className={`options-bar-img-button-container ${imageInput && hover ? "show-options-bar-img-buttons" : ""}`}>
                <img className="options-bar-img-button" src={expandIcon} onClick={expandDialog} style={{filter: getFilter()}} draggable={false}/>
                <img className="options-bar-img-button" src={segmentateIcon} onClick={segmentate} style={{filter: getFilter()}} draggable={false}/>
                <img className="options-bar-img-button" src={drawIcon} onClick={startDrawing} style={{filter: getFilter()}} draggable={false}/>
                <img className="options-bar-img-button" src={xIcon} onClick={removeImage} style={{filter: getFilter()}} draggable={false}/>
            </div>
            {imageInput ? 
            <canvas ref={ref} className="options-bar-img" draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%)`}}></canvas> :
            <img className="options-bar-img" src={imgPlaceHolder} style={{filter: getFilter()}} draggable={false}/>}
        </label>
        <input id="img" type="file" onChange={(event) => loadImage(event)}/>
        </>
    )
}

export default OptionsImage