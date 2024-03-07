import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, 
TrainTabContext, FolderLocationContext, SocketContext, TrainStartedContext, TrainProgressContext,
TrainProgressTextContext, TrainCompletedContext, TrainImagesContext, ModelNameContext, EpochsContext,
SaveEpochsContext, PreviewEpochsContext, PreviewPromptContext, LearningRateContext, GradientAccumulationStepsContext,
ResolutionContext, LearningFunctionContext, ImageBrightnessContext, ImageContrastContext, PreviewImageContext,
TrainRenderImageContext, TrainNameContext, ReverseSortContext} from "../Context"
import {ProgressBar, Dropdown, DropdownButton} from "react-bootstrap"
import xIcon from "../assets/icons/x.png"
import xIconHover from "../assets/icons/x-hover.png"
import functions from "../structures/Functions"
import folder from "../assets/icons/folder.png"
import TrainImage from "../components/TrainImage"
import "./styles/traintag.less"
import axios from "axios"

let timer = null as any
let clicking = false
let scrollLock = false

const TrainTextualInversion: React.FunctionComponent = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {imageBrightness, setImageBrightness} = useContext(ImageBrightnessContext)
    const {imageContrast, setImageContrast} = useContext(ImageContrastContext)
    const {socket, setSocket} = useContext(SocketContext)
    const {trainTab, setTrainTab} = useContext(TrainTabContext)
    const {folderLocation, setFolderLocation} = useContext(FolderLocationContext)
    const {trainImages, setTrainImages} = useContext(TrainImagesContext)
    const {trainProgress, setTrainProgress} = useContext(TrainProgressContext)
    const {trainProgressText, setTrainProgressText} = useContext(TrainProgressTextContext)
    const {trainStarted, setTrainStarted} = useContext(TrainStartedContext)
    const {trainCompleted, setTrainCompleted} = useContext(TrainCompletedContext)
    const {modelName, setModelName} = useContext(ModelNameContext)
    const [vectors, setVectors] = useState("1")
    const {epochs, setEpochs} = useContext(EpochsContext)
    const {saveEpochs, setSaveEpochs} = useContext(SaveEpochsContext)
    const {previewEpochs, setPreviewEpochs} = useContext(PreviewEpochsContext)
    const {previewPrompt, setPreviewPrompt} = useContext(PreviewPromptContext)
    const {learningRate, setLearningRate} = useContext(LearningRateContext)
    const {gradientAccumulationSteps, setGradientAccumulationSteps} = useContext(GradientAccumulationStepsContext)
    const {learningFunction, setLearningFunction} = useContext(LearningFunctionContext)
    const {resolution, setResolution} = useContext(ResolutionContext)
    const {previewImage, setPreviewImage} = useContext(PreviewImageContext)
    const {trainRenderImage, setTrainRenderImage} = useContext(TrainRenderImageContext)
    const {trainName, setTrainName} = useContext(TrainNameContext)
    const {reverseSort, setReverseSort} = useContext(ReverseSortContext)
    const [slice, setSlice] = useState([])
    const [sliceIndex, setSliceIndex] = useState(0)
    const [hover, setHover] = useState(false)
    const [xHover, setXHover] = useState(false)
    const progressBarRef = useRef(null) as React.RefObject<HTMLDivElement>
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const max = 100 + (sliceIndex * 100)
        let slice = reverseSort ? trainImages.slice(Math.max(trainImages.length - max - 1, 0), trainImages.length - 1) : trainImages.slice(0, max)
        setSlice(slice)
    }, [trainImages, reverseSort, sliceIndex])

    const handleScroll = (event: Event) => {
        if(!slice.length) return
        if (scrollLock) return
        if (Math.abs(document.body.scrollHeight - (document.body.scrollTop + document.body.clientHeight)) <= 1) {
            scrollLock = true
            setSliceIndex((prev: number) => prev + 1)
            setTimeout(() => {
                scrollLock = false
            }, 1000)
        }
    }

    useEffect(() => {
        document.body.addEventListener("scroll", handleScroll)
        return () => {
            document.body.removeEventListener("scroll", handleScroll)
        }
    }, [slice])

    const imagesJSX = () => {
        let jsx = [] as any
        if (reverseSort) {
            for (let i = slice.length - 1; i >= 0; i--) {
                jsx.push(<TrainImage img={trainImages[i]}/>)
            }
        } else {
            for (let i = 0; i < slice.length; i++) {
                jsx.push(<TrainImage img={trainImages[i]}/>)
            }
        }
        return jsx
    }

    useEffect(() => {
        const savedVectors = localStorage.getItem("vectors")
        if (savedVectors) setVectors(savedVectors)
    }, [])

    useEffect(() => {
        localStorage.setItem("vectors", String(vectors))
    }, [vectors])

    useEffect(() => {
        if (!socket) return
        const startTrain = () => {
            setTrainStarted(true)
            setTrainCompleted(false)
            setTrainProgress(-1)
            setTrainProgressText("")
        }
        const trackProgress = (data: any) => {
            const progress = (100 / Number(data.total_step)) * Number(data.step)
            setTrainStarted(true)
            setTrainCompleted(false)
            setTrainProgress(progress)
            setTrainProgressText(`Step ${data.step} / ${data.total_step} Epoch ${data.epoch} / ${data.total_epoch}`)
        }
        const completeTrain = async (data: any) => {
            setTrainCompleted(true)
            setTrainStarted(false)
        }
        const interruptTrain = () => {
            setTrainStarted(false)
            setTrainRenderImage("")
        }
        const trainImageProgress = (data: any) => {
            const pixels = new Uint8Array(data.image)
            const canvas = document.createElement("canvas")
            canvas.width = data.width
            canvas.height = data.height
            const ctx = canvas.getContext("2d")
            const newImageData = new ImageData(data.width, data.height)
            newImageData.data.set(pixels)
            ctx?.putImageData(newImageData, 0, 0)
            const url = canvas.toDataURL()
            setTrainRenderImage(url)
        }
        const trainImageComplete = (data: any) => {
            setTrainRenderImage(data.image ? `/retrieve?path=${data.image}` : "")
        }
        socket.on("train starting", startTrain)
        socket.on("train progress", trackProgress)
        socket.on("train complete", completeTrain)
        socket.on("train interrupt", interruptTrain)
        socket.on("train image progress", trainImageProgress)
        socket.on("train image complete", trainImageComplete)
        return () => {
            socket.off("train starting", startTrain)
            socket.off("train progress", trackProgress)
            socket.off("train complete", completeTrain)
            socket.off("train interrupt", interruptTrain)
            socket.off("train image progress", trainImageProgress)
            socket.off("train image complete", trainImageComplete)
        }
    }, [socket])

    const updateLocation = async () => {
        const location = await axios.post("/update-location").then((r) => r.data)
        if (location) setFolderLocation(location)
    }

    useEffect(() => {
        const updateTrainImages = async () => {
            let images = await axios.post("/list-files", {folder: folderLocation}).then((r) => r.data)
            if (images?.length) {
                images = images.map((i: string) => `/retrieve?path=${i}&?v=${new Date().getTime()}`)
                setTrainImages(images)
            }
        }
        updateTrainImages()
    }, [folderLocation])

    const getText = () => {
        if (trainCompleted) return "Completed"
        if (trainProgress >= 0) return trainProgressText
        return "Starting"
    }

    const getProgress = () => {
        if (trainCompleted) return 100
        if (trainProgress >= 0) return trainProgress
        return 0
    }

    const preview = () => {
        if (!trainCompleted || !trainRenderImage) return
        setPreviewImage(trainRenderImage)
    }

    const showInFolder = () => {
        if (!trainCompleted && !trainRenderImage) return
        axios.post("/show-in-folder", {path: trainRenderImage})
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

    const remove = () => {
        if (!trainCompleted) return
        setTrainStarted(false)
        setTrainRenderImage("")
    }

    const train = async () => {
        const json = {} as any
        json.images = trainImages.map((i: string) => i.replace("/retrieve?path=", ""))
        json.model_name = modelName
        json.train_data = folderLocation
        json.token = trainName
        json.num_vectors = Number(vectors)
        json.num_train_epochs = Number(epochs)
        json.learning_rate = Number(learningRate)
        json.gradient_accumulation_steps = Number(gradientAccumulationSteps)
        json.resolution = Number(resolution)
        json.save_epochs = Number(saveEpochs)
        json.validation_epochs = Number(previewEpochs)
        json.validation_prompt = previewPrompt
        json.learning_function = learningFunction
        await axios.post("/train-textual-inversion", json)
    }

    const interruptTrain = async () => {
        axios.post("/interrupt-train")
    }

    const openImageLocation = async () => {
        await axios.post("/open-folder", {absolute: folderLocation})
    }

    const openFolder = async () => {
        await axios.post("/open-folder", {path: `outputs/models/textual inversion/${trainName}`})
    }

    const reset = () => {
        setTrainName("")
        setVectors("1")
        setEpochs("20")
        setSaveEpochs("5")
        setPreviewEpochs("5")
        setPreviewPrompt("")
        setLearningRate("1e-4")
        setGradientAccumulationSteps("1")
        setResolution("256")
        setLearningFunction("constant")
    }

    return (
        <div className="train-tag" onMouseEnter={() => setEnableDrag(false)}>
            <div className="train-tag-folder-container">
                <img className="train-tag-folder" src={folder} style={{filter: getFilter()}} onClick={updateLocation}/>
                <div className="train-tag-location" onDoubleClick={openImageLocation}>{folderLocation ? folderLocation : "None"}</div>
                <button className="train-tag-button" onClick={() => trainStarted ? interruptTrain() : train()} style={{backgroundColor: trainStarted ? "var(--buttonBGStop)" : "var(--buttonBG)"}}>{trainStarted ? "Stop" : "Train"}</button>
                <button className="train-tag-button" onClick={() => openFolder()}>Open</button>
                <button className="train-tag-button" onClick={() => reset()}>Reset</button>
            </div>
            <div className="train-tag-settings-container">
                <div className="train-tag-settings-column">
                    <div className="train-tag-settings-box">
                        <span className="train-tag-settings-title">Name:</span>
                        <input className="train-tag-settings-input" type="text" spellCheck={false} value={trainName} onChange={(event) => setTrainName(event.target.value)}/>
                    </div>
                    <div className="train-tag-settings-box">
                        <span className="train-tag-settings-title">Vectors:</span>
                        <input className="train-tag-settings-input" type="text" spellCheck={false} value={vectors} onChange={(event) => setVectors(event.target.value)}/>
                    </div>
                    <div className="train-tag-settings-box">
                        <span className="train-tag-settings-title">Epochs:</span>
                        <input className="train-tag-settings-input" type="text" spellCheck={false} value={epochs} onChange={(event) => setEpochs(event.target.value)}/>
                    </div>
                    <div className="train-tag-settings-box">
                        <span className="train-tag-settings-title">Learning Rate:</span>
                        <input className="train-tag-settings-input" type="text" spellCheck={false} value={learningRate} onChange={(event) => setLearningRate(event.target.value)}/>
                    </div>
                    <div className="train-tag-settings-box">
                        <span className="train-tag-settings-title">Gradient Accumulation Steps:</span>
                        <input className="train-tag-settings-input" type="text" spellCheck={false} value={gradientAccumulationSteps} onChange={(event) => setGradientAccumulationSteps(event.target.value)}/>
                    </div>
                </div>
                <div className="train-tag-settings-column">
                    <div className="train-tag-settings-box">
                        <span className="train-tag-settings-title">Resolution:</span>
                        <input className="train-tag-settings-input" type="text" spellCheck={false} value={resolution} onChange={(event) => setResolution(event.target.value)}/>
                    </div>
                    <div className="train-tag-settings-box">
                        <span className="train-tag-settings-title">Save Epochs:</span>
                        <input className="train-tag-settings-input" type="text" spellCheck={false} value={saveEpochs} onChange={(event) => setSaveEpochs(event.target.value)}/>
                    </div>
                    <div className="train-tag-settings-box">
                        <span className="train-tag-settings-title">Preview Epochs:</span>
                        <input className="train-tag-settings-input" type="text" spellCheck={false} value={previewEpochs} onChange={(event) => setPreviewEpochs(event.target.value)}/>
                    </div>
                    <div className="train-tag-settings-box">
                        <span className="train-tag-settings-title">Preview Prompt:</span>
                        <input className="train-tag-settings-input" type="text" spellCheck={false} value={previewPrompt} onChange={(event) => setPreviewPrompt(event.target.value)}/>
                    </div>
                    <div className="train-tag-settings-box">
                        <span className="train-tag-settings-title">Learning Function:</span>
                        <DropdownButton title={learningFunction} drop="down" className="checkpoint-selector">
                            <Dropdown.Item active={learningFunction === "constant"} onClick={() => setLearningFunction("constant")}>constant</Dropdown.Item>
                            <Dropdown.Item active={learningFunction === "linear"} onClick={() => setLearningFunction("linear")}>linear</Dropdown.Item>
                            <Dropdown.Item active={learningFunction === "cosine"} onClick={() => setLearningFunction("cosine")}>cosine</Dropdown.Item>
                            <Dropdown.Item active={learningFunction === "quadratic"} onClick={() => setLearningFunction("quadratic")}>quadratic</Dropdown.Item>
                            <Dropdown.Item active={learningFunction === "cubic"} onClick={() => setLearningFunction("cubic")}>cubic</Dropdown.Item>
                            <Dropdown.Item active={learningFunction === "quartic"} onClick={() => setLearningFunction("quartic")}>quartic</Dropdown.Item>
                        </DropdownButton>
                    </div>
                </div>
            </div>
            {trainStarted ? <div className="train-tag-progress">
                <div className="render-progress-container" style={{filter: getFilter()}}>
                    <span className="render-progress-text">{getText()}</span>
                    <ProgressBar ref={progressBarRef} animated now={getProgress()}/>
                </div>
            </div> : null}
            {trainRenderImage ? <div className="render-img-block-container"><div className="render-img-container" onClick={handleClick} onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
                {trainCompleted ? <div className={`render-img-button-container ${hover ? "render-buttons-show" : ""}`}>
                    <img className="render-img-button" src={xHover ? xIconHover : xIcon} style={{filter: getFilter()}}
                    onMouseEnter={() => setXHover(true)} onMouseLeave={() => setXHover(false)} onClick={remove}/>
                </div> : null}
                <img className="render-img" src={trainRenderImage} draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%)`}}/>
            </div></div> : null}
            <div className="train-tag-images-container">
                {imagesJSX()}
            </div>
        </div>
    )
}

export default TrainTextualInversion